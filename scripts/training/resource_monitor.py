#!/usr/bin/env python
"""
Resource Monitor for ML Training

This module provides comprehensive system resource monitoring for machine learning training,
with special focus on M1 Mac optimization. It tracks CPU, memory, GPU, disk usage, thermal
state, and training metrics to provide early warnings, adaptive resource management, and
detailed logging for performance optimization.

Features:
- Real-time monitoring of CPU, memory, disk, and GPU usage
- Thermal state and throttling detection for M1 Macs
- Training throughput and performance metrics tracking
- Resource constraint alerts and early warning system
- W&B integration for visualization and logging
- Adaptive resource management (auto batch size adjustment)
- Memory leak detection and cleanup recommendations
- Network usage monitoring for distributed training
- System health scores and training efficiency metrics
- Crash prediction and prevention

Usage:
    from scripts.training.resource_monitor import ResourceMonitor
    
    # Initialize monitor
    monitor = ResourceMonitor(
        model=model,
        log_dir="logs/run_1",
        use_wandb=True,
        check_interval_seconds=5,
        alert_threshold=0.9,
    )
    
    # Start monitoring
    monitor.start()
    
    # During training
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Training step...
            loss = model(batch)
            
            # Update metrics
            monitor.update_training_metrics(
                step=step,
                loss=loss.item(),
                learning_rate=scheduler.get_last_lr()[0],
                batch_size=len(batch),
                throughput=images_per_second,
            )
    
    # Stop monitoring
    monitor.stop()
    
    # Get resource usage summary
    summary = monitor.get_summary()
    print(summary)
"""

import atexit
import datetime
import gc
import json
import logging
import os
import platform
import psutil
import re
import shutil
import signal
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("W&B not available. Install with: pip install wandb")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logger.warning("GPUtil not available. Install with: pip install gputil")

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class ResourceType(Enum):
    """Types of resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    THERMAL = "thermal"
    TRAINING = "training"
    SYSTEM = "system"


class AlertLevel(Enum):
    """Alert levels for resource monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SystemPlatform(Enum):
    """Supported system platforms."""
    UNKNOWN = "unknown"
    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"
    MACOS_SILICON = "macos_silicon"  # Apple Silicon (M1/M2)


@dataclass
class ResourceMetrics:
    """Base class for resource metrics."""
    timestamp: float = field(default_factory=time.time)
    resource_type: ResourceType = ResourceType.SYSTEM


@dataclass
class CPUMetrics(ResourceMetrics):
    """CPU usage metrics."""
    usage_percent: float = 0.0
    usage_per_core: List[float] = field(default_factory=list)
    load_avg_1min: float = 0.0
    load_avg_5min: float = 0.0
    load_avg_15min: float = 0.0
    context_switches: int = 0
    interrupts: int = 0
    soft_interrupts: int = 0
    frequency_mhz: float = 0.0
    
    def __post_init__(self):
        self.resource_type = ResourceType.CPU


@dataclass
class MemoryMetrics(ResourceMetrics):
    """Memory usage metrics."""
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    available_gb: float = 0.0
    percent_used: float = 0.0
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    swap_free_gb: float = 0.0
    swap_percent: float = 0.0
    page_faults: int = 0
    pytorch_allocated_gb: float = 0.0
    pytorch_reserved_gb: float = 0.0
    pytorch_active_gb: float = 0.0
    
    def __post_init__(self):
        self.resource_type = ResourceType.MEMORY


@dataclass
class GPUMetrics(ResourceMetrics):
    """GPU usage metrics."""
    device_count: int = 0
    devices: List[Dict[str, Any]] = field(default_factory=list)
    total_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    utilization_percent: float = 0.0
    temperature_c: float = 0.0
    power_usage_watts: float = 0.0
    power_limit_watts: float = 0.0
    memory_temperature_c: float = 0.0
    is_mps_available: bool = False
    
    def __post_init__(self):
        self.resource_type = ResourceType.GPU


@dataclass
class DiskMetrics(ResourceMetrics):
    """Disk usage metrics."""
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    percent_used: float = 0.0
    read_count: int = 0
    write_count: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    read_time_ms: int = 0
    write_time_ms: int = 0
    
    def __post_init__(self):
        self.resource_type = ResourceType.DISK


@dataclass
class NetworkMetrics(ResourceMetrics):
    """Network usage metrics."""
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    errin: int = 0
    errout: int = 0
    dropin: int = 0
    dropout: int = 0
    bandwidth_mbps: float = 0.0
    
    def __post_init__(self):
        self.resource_type = ResourceType.NETWORK


@dataclass
class ThermalMetrics(ResourceMetrics):
    """Thermal state metrics."""
    cpu_temperature_c: float = 0.0
    gpu_temperature_c: float = 0.0
    is_throttling: bool = False
    fan_speed_rpm: int = 0
    thermal_pressure: str = "nominal"  # nominal, moderate, heavy, critical
    
    def __post_init__(self):
        self.resource_type = ResourceType.THERMAL


@dataclass
class TrainingMetrics(ResourceMetrics):
    """Training performance metrics."""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    batch_size: int = 0
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    optimizer_time_ms: float = 0.0
    total_time_ms: float = 0.0
    grad_norm: float = 0.0
    
    def __post_init__(self):
        self.resource_type = ResourceType.TRAINING


@dataclass
class SystemHealthMetrics(ResourceMetrics):
    """System health and efficiency metrics."""
    health_score: float = 1.0  # 0.0 (bad) to 1.0 (good)
    cpu_efficiency: float = 1.0
    memory_efficiency: float = 1.0
    gpu_efficiency: float = 1.0
    io_efficiency: float = 1.0
    training_efficiency: float = 1.0
    bottleneck: str = "none"  # cpu, memory, gpu, disk, network, thermal
    crash_probability: float = 0.0
    memory_leak_detected: bool = False
    throttling_detected: bool = False
    io_bottleneck_detected: bool = False
    
    def __post_init__(self):
        self.resource_type = ResourceType.SYSTEM


@dataclass
class ResourceAlert:
    """Alert for resource issues."""
    timestamp: float = field(default_factory=time.time)
    resource_type: ResourceType = ResourceType.SYSTEM
    level: AlertLevel = AlertLevel.INFO
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    suggested_actions: List[str] = field(default_factory=list)


class ResourceMonitor:
    """
    Comprehensive system resource monitor for ML training.
    
    This class provides real-time monitoring of system resources during
    training, with special focus on M1 Mac optimization. It tracks CPU,
    memory, GPU, disk usage, thermal state, and training metrics to provide
    early warnings, adaptive resource management, and detailed logging.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        log_dir: str = "logs/resource_monitor",
        use_wandb: bool = False,
        project_name: str = "ml-training",
        check_interval_seconds: float = 5.0,
        alert_threshold: float = 0.8,
        critical_threshold: float = 0.95,
        enable_thermal_monitoring: bool = True,
        enable_memory_leak_detection: bool = True,
        enable_crash_prediction: bool = True,
        enable_adaptive_resources: bool = False,
        log_to_file: bool = True,
        log_to_console: bool = True,
        alert_callbacks: Optional[Dict[AlertLevel, Callable]] = None,
        monitor_gpu: bool = True,
        monitor_network: bool = True,
        monitor_disk: bool = True,
        max_history_size: int = 1000,
    ):
        """
        Initialize the resource monitor.
        
        Args:
            model: PyTorch model to monitor (optional)
            log_dir: Directory to save logs
            use_wandb: Whether to log to W&B
            project_name: W&B project name
            check_interval_seconds: Interval between resource checks
            alert_threshold: Threshold for warning alerts (0.0-1.0)
            critical_threshold: Threshold for critical alerts (0.0-1.0)
            enable_thermal_monitoring: Whether to monitor thermal state
            enable_memory_leak_detection: Whether to detect memory leaks
            enable_crash_prediction: Whether to predict potential crashes
            enable_adaptive_resources: Whether to adapt resources automatically
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            alert_callbacks: Callbacks for different alert levels
            monitor_gpu: Whether to monitor GPU
            monitor_network: Whether to monitor network
            monitor_disk: Whether to monitor disk
            max_history_size: Maximum number of metrics to keep in history
        """
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Monitoring settings
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.project_name = project_name
        self.check_interval_seconds = check_interval_seconds
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.enable_thermal_monitoring = enable_thermal_monitoring
        self.enable_memory_leak_detection = enable_memory_leak_detection
        self.enable_crash_prediction = enable_crash_prediction
        self.enable_adaptive_resources = enable_adaptive_resources
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.alert_callbacks = alert_callbacks or {}
        self.monitor_gpu = monitor_gpu
        self.monitor_network = monitor_network
        self.monitor_disk = monitor_disk
        self.max_history_size = max_history_size
        
        # Determine system platform
        self.platform = self._detect_platform()
        logger.info(f"Detected platform: {self.platform.value}")
        
        # Initialize metrics history
        self.cpu_metrics_history: List[CPUMetrics] = []
        self.memory_metrics_history: List[MemoryMetrics] = []
        self.gpu_metrics_history: List[GPUMetrics] = []
        self.disk_metrics_history: List[DiskMetrics] = []
        self.network_metrics_history: List[NetworkMetrics] = []
        self.thermal_metrics_history: List[ThermalMetrics] = []
        self.training_metrics_history: List[TrainingMetrics] = []
        self.system_health_history: List[SystemHealthMetrics] = []
        self.alerts_history: List[ResourceAlert] = []
        
        # Initialize state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.start_time = None
        self.last_check_time = None
        self.last_network_metrics = None
        self.last_disk_metrics = None
        self.memory_baseline = None
        self.potential_memory_leaks = []
        self.recommended_batch_size = None
        
        # Initialize log file
        if self.log_to_file:
            self.log_file_path = self.log_dir / "resource_monitor.log"
            self.metrics_file_path = self.log_dir / "resource_metrics.jsonl"
            self.alerts_file_path = self.log_dir / "resource_alerts.jsonl"
        
        # Initialize W&B
        if self.use_wandb:
            if not wandb.run:
                wandb.init(project=self.project_name, config={
                    "monitor_interval": self.check_interval_seconds,
                    "platform": self.platform.value,
                })
        
        # Register signal handlers and exit handlers
        self._register_handlers()
        
        logger.info("Resource monitor initialized")
    
    def _detect_platform(self) -> SystemPlatform:
        """
        Detect the system platform.
        
        Returns:
            SystemPlatform: Detected platform
        """
        system = platform.system().lower()
        
        if system == "darwin":
            # Check if running on Apple Silicon
            try:
                output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                if "apple" in output.lower():
                    return SystemPlatform.MACOS_SILICON
                else:
                    return SystemPlatform.MACOS
            except Exception:
                # Fall back to generic macOS if detection fails
                return SystemPlatform.MACOS
        elif system == "linux":
            return SystemPlatform.LINUX
        elif system == "windows":
            return SystemPlatform.WINDOWS
        else:
            return SystemPlatform.UNKNOWN
    
    def _register_handlers(self):
        """Register signal and exit handlers."""
        # Register exit handler
        atexit.register(self.stop)
        
        # Register signal handlers
        def signal_handler(sig, frame):
            logger.warning(f"Received signal {sig}, stopping resource monitor...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """Start the resource monitoring."""
        if self.is_monitoring:
            logger.warning("Resource monitor is already running")
            return
        
        self.is_monitoring = True
        self.start_time = time.time()
        self.last_check_time = self.start_time
        
        # Create initial baseline measurements
        self._collect_all_metrics()
        self.memory_baseline = self._get_latest_memory_metrics()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Resource monitoring started")
        
        # Log initial system info
        self._log_system_info()
    
    def stop(self):
        """Stop the resource monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread is not None:
            self.monitoring_thread.join(timeout=2.0)
            self.monitoring_thread = None
        
        # Generate final summary
        summary = self.get_summary()
        
        # Log final summary
        if self.log_to_console:
            logger.info("Resource monitoring stopped")
            logger.info(f"Monitoring duration: {summary['duration_formatted']}")
            logger.info(f"Average CPU usage: {summary['cpu']['average_usage_percent']:.1f}%")
            logger.info(f"Average memory usage: {summary['memory']['average_percent_used']:.1f}%")
            if self.monitor_gpu and summary['gpu']['device_count'] > 0:
                logger.info(f"Average GPU usage: {summary['gpu']['average_utilization_percent']:.1f}%")
            logger.info(f"Total alerts: {summary['alerts']['total']}")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({"resource_monitor/summary": summary})
    
    def update_training_metrics(
        self,
        step: int,
        epoch: int = 0,
        loss: float = 0.0,
        learning_rate: float = 0.0,
        batch_size: int = 0,
        samples_per_second: float = 0.0,
        tokens_per_second: float = 0.0,
        forward_time_ms: float = 0.0,
        backward_time_ms: float = 0.0,
        optimizer_time_ms: float = 0.0,
        total_time_ms: float = 0.0,
        grad_norm: float = 0.0,
    ):
        """
        Update training metrics.
        
        Args:
            step: Current training step
            epoch: Current epoch
            loss: Training loss
            learning_rate: Current learning rate
            batch_size: Batch size
            samples_per_second: Training throughput in samples/second
            tokens_per_second: Training throughput in tokens/second
            forward_time_ms: Time for forward pass in ms
            backward_time_ms: Time for backward pass in ms
            optimizer_time_ms: Time for optimizer step in ms
            total_time_ms: Total time for step in ms
            grad_norm: Gradient norm
        """
        metrics = TrainingMetrics(
            timestamp=time.time(),
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            batch_size=batch_size,
            samples_per_second=samples_per_second,
            tokens_per_second=tokens_per_second,
            forward_time_ms=forward_time_ms,
            backward_time_ms=backward_time_ms,
            optimizer_time_ms=optimizer_time_ms,
            total_time_ms=total_time_ms,
            grad_norm=grad_norm,
        )
        
        # Add to history
        self._add_to_history(self.training_metrics_history, metrics)
        
        # Log metrics
        self._log_metrics(metrics)
        
        # Check for training issues
        self._check_training_health(metrics)
    
    def get_recommended_batch_size(self) -> Optional[int]:
        """
        Get the recommended batch size based on resource usage.
        
        Returns:
            Optional[int]: Recommended batch size or None if not available
        """
        return self.recommended_batch_size
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of resource usage.
        
        Returns:
            Dict[str, Any]: Resource usage summary
        """
        # Calculate duration
        end_time = time.time()
        duration_seconds = end_time - self.start_time
        duration_formatted = str(datetime.timedelta(seconds=int(duration_seconds)))
        
        # CPU summary
        cpu_metrics = self.cpu_metrics_history
        cpu_summary = {
            "average_usage_percent": np.mean([m.usage_percent for m in cpu_metrics]) if cpu_metrics else 0.0,
            "max_usage_percent": np.max([m.usage_percent for m in cpu_metrics]) if cpu_metrics else 0.0,
            "average_load_1min": np.mean([m.load_avg_1min for m in cpu_metrics]) if cpu_metrics else 0.0,
        }
        
        # Memory summary
        memory_metrics = self.memory_metrics_history
        memory_summary = {
            "average_percent_used": np.mean([m.percent_used for m in memory_metrics]) if memory_metrics else 0.0,
            "max_percent_used": np.max([m.percent_used for m in memory_metrics]) if memory_metrics else 0.0,
            "average_used_gb": np.mean([m.used_gb for m in memory_metrics]) if memory_metrics else 0.0,
            "max_used_gb": np.max([m.used_gb for m in memory_metrics]) if memory_metrics else 0.0,
            "total_gb": memory_metrics[-1].total_gb if memory_metrics else 0.0,
        }
        
        # GPU summary
        gpu_metrics = self.gpu_metrics_history
        gpu_summary = {
            "device_count": gpu_metrics[-1].device_count if gpu_metrics else 0,
            "average_utilization_percent": np.mean([m.utilization_percent for m in gpu_metrics]) if gpu_metrics else 0.0,
            "max_utilization_percent": np.max([m.utilization_percent for m in gpu_metrics]) if gpu_metrics else 0.0,
            "average_used_memory_gb": np.mean([m.used_memory_gb for m in gpu_metrics]) if gpu_metrics else 0.0,
            "max_used_memory_gb": np.max([m.used_memory_gb for m in gpu_metrics]) if gpu_metrics else 0.0,
            "average_temperature_c": np.mean([m.temperature_c for m in gpu_metrics]) if gpu_metrics else 0.0,
            "max_temperature_c": np.max([m.temperature_c for m in gpu_metrics]) if gpu_metrics else 0.0,
        }
        
        # Disk summary
        disk_metrics = self.disk_metrics_history
        disk_summary = {
            "average_percent_used": np.mean([m.percent_used for m in disk_metrics]) if disk_metrics else 0.0,
            "free_gb": disk_metrics[-1].free_gb if disk_metrics else 0.0,
            "total_gb": disk_metrics[-1].total_gb if disk_metrics else 0.0,
        }
        
        # Network summary
        network_metrics = self.network_metrics_history
        network_summary = {
            "total_bytes_sent": network_metrics[-1].bytes_sent if network_metrics else 0,
            "total_bytes_recv": network_metrics[-1].bytes_recv if network_metrics else 0,
            "average_bandwidth_mbps": np.mean([m.bandwidth_mbps for m in network_metrics]) if network_metrics else 0.0,
        }
        
        # Thermal summary
        thermal_metrics = self.thermal_metrics_history
        thermal_summary = {
            "average_cpu_temperature_c": np.mean([m.cpu_temperature_c for m in thermal_metrics]) if thermal_metrics else 0.0,
            "max_cpu_temperature_c": np.max([m.cpu_temperature_c for m in thermal_metrics]) if thermal_metrics else 0.0,
            "throttling_detected": any(m.is_throttling for m in thermal_metrics) if thermal_metrics else False,
        }
        
        # Training summary
        training_metrics = self.training_metrics_history
        training_summary = {
            "last_step": training_metrics[-1].step if training_metrics else 0,
            "last_epoch": training_metrics[-1].epoch if training_metrics else 0,
            "average_loss": np.mean([m.loss for m in training_metrics]) if training_metrics else 0.0,
            "average_samples_per_second": np.mean([m.samples_per_second for m in training_metrics]) if training_metrics else 0.0,
            "average_tokens_per_second": np.mean([m.tokens_per_second for m in training_metrics]) if training_metrics else 0.0,
            "average_batch_size": np.mean([m.batch_size for m in training_metrics]) if training_metrics else 0,
        }
        
        # System health summary
        health_metrics = self.system_health_history
        health_summary = {
            "average_health_score": np.mean([m.health_score for m in health_metrics]) if health_metrics else 1.0,
            "min_health_score": np.min([m.health_score for m in health_metrics]) if health_metrics else 1.0,
            "bottlenecks": list(set(m.bottleneck for m in health_metrics if m.bottleneck != "none")),
            "memory_leak_detected": any(m.memory_leak_detected for m in health_metrics) if health_metrics else False,
            "throttling_detected": any(m.throttling_detected for m in health_metrics) if health_metrics else False,
        }
        
        # Alerts summary
        alerts = self.alerts_history
        alerts_summary = {
            "total": len(alerts),
            "info": len([a for a in alerts if a.level == AlertLevel.INFO]),
            "warning": len([a for a in alerts if a.level == AlertLevel.WARNING]),
            "critical": len([a for a in alerts if a.level == AlertLevel.CRITICAL]),
            "emergency": len([a for a in alerts if a.level == AlertLevel.EMERGENCY]),
        }
        
        return {
            "duration_seconds": duration_seconds,
            "duration_formatted": duration_formatted,
            "start_time": self.start_time,
            "end_time": end_time,
            "platform": self.platform.value,
            "cpu": cpu_summary,
            "memory": memory_summary,
            "gpu": gpu_summary,
            "disk": disk_summary,
            "network": network_summary,
            "thermal": thermal_summary,
            "training": training_summary,
            "system_health": health_summary,
            "alerts": alerts_summary,
            "recommended_batch_size": self.recommended_batch_size,
        }
    
    def get_latest_metrics(self, resource_type: ResourceType) -> Optional[ResourceMetrics]:
        """
        Get the latest metrics for a specific resource type.
        
        Args:
            resource_type: Type of resource
            
        Returns:
            Optional[ResourceMetrics]: Latest metrics or None if not available
        """
        if resource_type == ResourceType.CPU and self.cpu_metrics_history:
            return self.cpu_metrics_history[-1]
        elif resource_type == ResourceType.MEMORY and self.memory_metrics_history:
            return self.memory_metrics_history[-1]
        elif resource_type == ResourceType.GPU and self.gpu_metrics_history:
            return self.gpu_metrics_history[-1]
        elif resource_type == ResourceType.DISK and self.disk_metrics_history:
            return self.disk_metrics_history[-1]
        elif resource_type == ResourceType.NETWORK and self.network_metrics_history:
            return self.network_metrics_history[-1]
        elif resource_type == ResourceType.THERMAL and self.thermal_metrics_history:
            return self.thermal_metrics_history[-1]
        elif resource_type == ResourceType.TRAINING and self.training_metrics_history:
            return self.training_metrics_history[-1]
        elif resource_type == ResourceType.SYSTEM and self.system_health_history:
            return self.system_health_history[-1]
        else:
            return None
    
    def get_memory_leak_info(self) -> Dict[str, Any]:
        """
        Get information about detected memory leaks.
        
        Returns:
            Dict[str, Any]: Memory leak information
        """
        if not self.enable_memory_leak_detection:
            return {"enabled": False}
        
        memory_metrics = self.memory_metrics_history
        if not memory_metrics or len(memory_metrics) < 10:
            return {"enabled": True, "detected": False, "reason": "Not enough data"}
        
        # Calculate memory growth rate
        times = [m.timestamp - self.start_time for m in memory_metrics]
        memory_usage = [m.used_gb for m in memory_metrics]
        
        # Simple linear regression to detect trend
        if len(times) >= 2:
            slope, intercept = np.polyfit(times, memory_usage, 1)
            growth_rate_mb_per_hour = slope * 3600 * 1000  # Convert to MB/hour
            
            return {
                "enabled": True,
                "detected": len(self.potential_memory_leaks) > 0,
                "growth_rate_mb_per_hour": growth_rate_mb_per_hour,
                "potential_leaks": self.potential_memory_leaks,
                "baseline_memory_gb": self.memory_baseline.used_gb if self.memory_baseline else 0.0,
                "current_memory_gb": memory_metrics[-1].used_gb,
                "increase_gb": memory_metrics[-1].used_gb - (self.memory_baseline.used_gb if self.memory_baseline else 0.0),
                "pytorch_allocated_gb": memory_metrics[-1].pytorch_allocated_gb,
                "pytorch_reserved_gb": memory_metrics[-1].pytorch_reserved_gb,
            }
        
        return {"enabled": True, "detected": False, "reason": "Not enough data points for trend analysis"}
    
    def get_crash_prediction(self) -> Dict[str, Any]:
        """
        Get crash prediction information.
        
        Returns:
            Dict[str, Any]: Crash prediction information
        """
        if not self.enable_crash_prediction:
            return {"enabled": False}
        
        health_metrics = self.system_health_history
        if not health_metrics:
            return {"enabled": True, "crash_probability": 0.0, "reason": "Not enough data"}
        
        latest_health = health_metrics[-1]
        
        # Factors that contribute to crash risk
        risk_factors = []
        
        # Memory pressure
        memory_metrics = self._get_latest_memory_metrics()
        if memory_metrics and memory_metrics.percent_used > 95:
            risk_factors.append({"factor": "memory_pressure", "severity": "critical"})
        elif memory_metrics and memory_metrics.percent_used > 85:
            risk_factors.append({"factor": "memory_pressure", "severity": "high"})
        
        # GPU memory pressure (if applicable)
        gpu_metrics = self._get_latest_gpu_metrics()
        if gpu_metrics and gpu_metrics.device_count > 0:
            if gpu_metrics.used_memory_gb / gpu_metrics.total_memory_gb > 0.95:
                risk_factors.append({"factor": "gpu_memory_pressure", "severity": "critical"})
            elif gpu_metrics.used_memory_gb / gpu_metrics.total_memory_gb > 0.85:
                risk_factors.append({"factor": "gpu_memory_pressure", "severity": "high"})
        
        # Thermal throttling
        thermal_metrics = self._get_latest_thermal_metrics()
        if thermal_metrics and thermal_metrics.is_throttling:
            risk_factors.append({"factor": "thermal_throttling", "severity": "high"})
        
        # Memory leaks
        if self.enable_memory_leak_detection and len(self.potential_memory_leaks) > 0:
            risk_factors.append({"factor": "memory_leak", "severity": "medium"})
        
        # Disk space
        disk_metrics = self._get_latest_disk_metrics()
        if disk_metrics and disk_metrics.free_gb < 1.0:
            risk_factors.append({"factor": "disk_space", "severity": "critical"})
        elif disk_metrics and disk_metrics.free_gb < 5.0:
            risk_factors.append({"factor": "disk_space", "severity": "high"})
        
        # Calculate crash probability based on risk factors
        crash_probability = latest_health.crash_probability
        
        return {
            "enabled": True,
            "crash_probability": crash_probability,
            "risk_factors": risk_factors,
            "health_score": latest_health.health_score,
            "bottleneck": latest_health.bottleneck,
        }
    
    def get_adaptive_resource_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for adaptive resource management.
        
        Returns:
            Dict[str, Any]: Resource recommendations
        """
        if not self.enable_adaptive_resources:
            return {"enabled": False}
        
        recommendations = {
            "enabled": True,
            "batch_size": self.recommended_batch_size,
            "recommendations": [],
        }
        
        # Memory-based recommendations
        memory_metrics = self._get_latest_memory_metrics()
        if memory_metrics:
            if memory_metrics.percent_used > 90:
                recommendations["recommendations"].append({
                    "resource": "memory",
                    "action": "reduce_batch_size",
                    "reason": "High memory usage",
                    "current_value": memory_metrics.percent_used,
                    "suggested_change": "Reduce batch size by 25%",
                })
            elif memory_metrics.percent_used < 50:
                recommendations["recommendations"].append({
                    "resource": "memory",
                    "action": "increase_batch_size",
                    "reason": "Low memory usage",
                    "current_value": memory_metrics.percent_used,
                    "suggested_change": "Increase batch size by 20%",
                })
        
        # GPU-based recommendations
        gpu_metrics = self._get_latest_gpu_metrics()
        if gpu_metrics and gpu_metrics.device_count > 0:
            memory_utilization = gpu_metrics.used_memory_gb / gpu_metrics.total_memory_gb if gpu_metrics.total_memory_gb > 0 else 0
            if memory_utilization > 0.9:
                recommendations["recommendations"].append({
                    "resource": "gpu",
                    "action": "reduce_batch_size",
                    "reason": "High GPU memory usage",
                    "current_value": memory_utilization,
                    "suggested_change": "Reduce batch size by 25%",
                })
            elif memory_utilization < 0.5 and gpu_metrics.utilization_percent > 80:
                recommendations["recommendations"].append({
                    "resource": "gpu",
                    "action": "increase_batch_size",
                    "reason": "GPU is compute-bound but has available memory",
                    "current_value": memory_utilization,
                    "suggested_change": "Increase batch size by 20%",
                })
        
        # Thermal-based recommendations
        thermal_metrics = self._get_latest_thermal_metrics()
        if thermal_metrics and thermal_metrics.is_throttling:
            recommendations["recommendations"].append({
                "resource": "thermal",
                "action": "reduce_workload",
                "reason": "Thermal throttling detected",
                "current_value": thermal_metrics.cpu_temperature_c,
                "suggested_change": "Reduce batch size or take a break to cool down",
            })
        
        return recommendations
    
    def _monitoring_loop(self):
        """Main monitoring loop that collects metrics at regular intervals."""
        while self.is_monitoring:
            try:
                # Collect metrics
                self._collect_all_metrics()
                
                # Analyze system health
                self._analyze_system_health()
                
                # Check for memory leaks
                if self.enable_memory_leak_detection:
                    self._check_memory_leaks()
                
                # Update adaptive resource recommendations
                if self.enable_adaptive_resources:
                    self._update_adaptive_recommendations()
                
                # Update last check time
                self.last_check_time = time.time()
                
                # Sleep until next check
                time.sleep(self.check_interval_seconds)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.check_interval_seconds)
    
    def _collect_all_metrics(self):
        """Collect all resource metrics."""
        # CPU metrics
        cpu_metrics = self._collect_cpu_metrics()
        self._add_to_history(self.cpu_metrics_history, cpu_metrics)
        
        # Memory metrics
        memory_metrics = self._collect_memory_metrics()
        self._add_to_history(self.memory_metrics_history, memory_metrics)
        
        # GPU metrics (if enabled)
        if self.monitor_gpu:
            gpu_metrics = self._collect_gpu_metrics()
            self._add_to_history(self.gpu_metrics_history, gpu_metrics)
        
        # Disk metrics (if enabled)
        if self.monitor_disk:
            disk_metrics = self._collect_disk_metrics()
            self._add_to_history(self.disk_metrics_history, disk_metrics)
        
        # Network metrics (if enabled)
        if self.monitor_network:
            network_metrics = self._collect_network_metrics()
            self._add_to_history(self.network_metrics_history, network_metrics)
        
        # Thermal metrics (if enabled)
        if self.enable_thermal_monitoring:
            thermal_metrics = self._collect_thermal_metrics()
            self._add_to_history(self.thermal_metrics_history, thermal_metrics)
        
        # Log metrics
        self._log_metrics(cpu_metrics)
        self._log_metrics(memory_metrics)
        if self.monitor_gpu:
            self._log_metrics(gpu_metrics)
        if self.monitor_disk:
            self._log_metrics(disk_metrics)
        if self.monitor_network:
            self._log_metrics(network_metrics)
        if self.enable_thermal_monitoring:
            self._log_metrics(thermal_metrics)
    
    def _collect_cpu_metrics(self) -> CPUMetrics:
        """
        Collect CPU metrics.
        
        Returns:
            CPUMetrics: CPU usage metrics
        """
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_percent_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Get load averages
            if platform.system() != "Windows":
                load_avg = psutil.getloadavg()
                load_1min, load_5min, load_15min = load_avg
            else:
                # Windows doesn't support getloadavg
                load_1min = load_5min = load_15min = cpu_percent / 100.0
            
            # Get CPU stats
            cpu_stats = psutil.cpu_stats()
            ctx_switches = cpu_stats.ctx_switches
            interrupts = cpu_stats.interrupts
            soft_interrupts = getattr(cpu_stats, "soft_interrupts", 0)
            
            # Get CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                frequency_mhz = cpu_freq.current
            else:
                frequency_mhz = 0.0
            
            return CPUMetrics(
                usage_percent=cpu_percent,
                usage_per_core=cpu_percent_per_core,
                load_avg_1min=load_1min,
                load_avg_5min=load_5min,
                load_avg_15min=load_15min,
                context_switches=ctx_switches,
                interrupts=interrupts,
                soft_interrupts=soft_interrupts,
                frequency_mhz=frequency_mhz,
            )
        
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}")
            return CPUMetrics()
    
    def _collect_memory_metrics(self) -> MemoryMetrics:
        """
        Collect memory metrics.
        
        Returns:
            MemoryMetrics: Memory usage metrics
        """
        try:
            # Get virtual memory
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024 ** 3)
            used_gb = vm.used / (1024 ** 3)
            free_gb = vm.free / (1024 ** 3)
            available_gb = vm.available / (1024 ** 3)
            percent_used = vm.percent
            
            # Get swap memory
            swap = psutil.swap_memory()
            swap_total_gb = swap.total / (1024 ** 3)
            swap_used_gb = swap.used / (1024 ** 3)
            swap_free_gb = swap.free / (1024 ** 3)
            swap_percent = swap.percent
            
            # Get page faults (not available on all platforms)
            page_faults = 0
            
            # Get PyTorch memory stats
            pytorch_allocated_gb = 0.0
            pytorch_reserved_gb = 0.0
            pytorch_active_gb = 0.0
            
            if torch.cuda.is_available():
                pytorch_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                pytorch_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                # Active memory is not directly available
            
            elif hasattr(torch, 'mps') and torch.mps.is_available():
                # MPS doesn't have the same memory management API as CUDA
                # We can't directly measure MPS memory usage
                pass
            
            return MemoryMetrics(
                total_gb=total_gb,
                used_gb=used_gb,
                free_gb=free_gb,
                available_gb=available_gb,
                percent_used=percent_used,
                swap_total_gb=swap_total_gb,
                swap_used_gb=swap_used_gb,
                swap_free_gb=swap_free_gb,
                swap_percent=swap_percent,
                page_faults=page_faults,
                pytorch_allocated_gb=pytorch_allocated_gb,
                pytorch_reserved_gb=pytorch_reserved_gb,
                pytorch_active_gb=pytorch_active_gb,
            )
        
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
            return MemoryMetrics()
    
    def _collect_gpu_metrics(self) -> GPUMetrics:
        """
        Collect GPU metrics.
        
        Returns:
            GPUMetrics: GPU usage metrics
        """
        try:
            # Initialize metrics
            device_count = 0
            devices = []
            total_memory_gb = 0.0
            used_memory_gb = 0.0
            free_memory_gb = 0.0
            utilization_percent = 0.0
            temperature_c = 0.0
            power_usage_watts = 0.0
            power_limit_watts = 0.0
            memory_temperature_c = 0.0
            is_mps_available = False
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch, 'mps') and torch.mps.is_available():
                is_mps_available = True
                device_count = 1
                
                # MPS doesn't provide detailed metrics like CUDA
                # Use system metrics as a proxy
                
                # For Apple Silicon, we can try to get GPU info from system_profiler
                if self.platform == SystemPlatform.MACOS_SILICON:
                    try:
                        # Get memory info from system
                        vm = psutil.virtual_memory()
                        total_memory_gb = vm.total / (1024 ** 3)
                        
                        # This is a rough approximation - unified memory makes it hard to separate
                        used_memory_gb = vm.used / (1024 ** 3) * 0.3  # Assume 30% is GPU usage
                        free_memory_gb = total_memory_gb - used_memory_gb
                        
                        # Try to get GPU utilization and temperature
                        # This requires additional tools that may not be available
                        
                        devices.append({
                            "name": "Apple Silicon GPU",
                            "memory_total_gb": total_memory_gb,
                            "memory_used_gb": used_memory_gb,
                            "memory_free_gb": free_memory_gb,
                            "utilization_percent": utilization_percent,
                            "temperature_c": temperature_c,
                        })
                    except Exception as e:
                        logger.debug(f"Error getting Apple Silicon GPU metrics: {e}")
            
            # Check for CUDA
            elif torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                
                # Get metrics for each device
                for i in range(device_count):
                    device_total_memory = torch.cuda.get_device_properties(i).total_memory
                    device_total_memory_gb = device_total_memory / (1024 ** 3)
                    device_allocated_memory = torch.cuda.memory_allocated(i)
                    device_allocated_memory_gb = device_allocated_memory / (1024 ** 3)
                    device_free_memory_gb = device_total_memory_gb - device_allocated_memory_gb
                    
                    device_info = {
                        "name": torch.cuda.get_device_name(i),
                        "memory_total_gb": device_total_memory_gb,
                        "memory_used_gb": device_allocated_memory_gb,
                        "memory_free_gb": device_free_memory_gb,
                        "utilization_percent": 0.0,  # Not directly available from PyTorch
                        "temperature_c": 0.0,  # Not directly available from PyTorch
                    }
                    
                    # Try to get additional metrics from nvidia-smi via GPUtil
                    if GPUTIL_AVAILABLE:
                        try:
                            gpus = GPUtil.getGPUs()
                            if i < len(gpus):
                                gpu = gpus[i]
                                device_info["utilization_percent"] = gpu.load * 100
                                device_info["temperature_c"] = gpu.temperature
                        except Exception as e:
                            logger.debug(f"Error getting GPUtil metrics: {e}")
                    
                    # Try to get additional metrics from NVML
                    if NVML_AVAILABLE:
                        try:
                            nvml.nvmlInit()
                            handle = nvml.nvmlDeviceGetHandleByIndex(i)
                            
                            # Get utilization
                            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                            device_info["utilization_percent"] = utilization.gpu
                            
                            # Get temperature
                            device_info["temperature_c"] = nvml.nvmlDeviceGetTemperature(
                                handle, nvml.NVML_TEMPERATURE_GPU
                            )
                            
                            # Get power usage
                            power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert from mW to W
                            device_info["power_usage_watts"] = power_usage
                            
                            # Get power limit
                            power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # Convert from mW to W
                            device_info["power_limit_watts"] = power_limit
                            
                            nvml.nvmlShutdown()
                        except Exception as e:
                            logger.debug(f"Error getting NVML metrics: {e}")
                    
                    devices.append(device_info)
                    
                    # Accumulate totals
                    total_memory_gb += device_total_memory_gb
                    used_memory_gb += device_allocated_memory_gb
                    free_memory_gb += device_free_memory_gb
                    utilization_percent += device_info["utilization_percent"]
                    temperature_c += device_info["temperature_c"]
                    power_usage_watts += device_info.get("power_usage_watts", 0.0)
                    power_limit_watts += device_info.get("power_limit_watts", 0.0)
                
                # Calculate averages
                if device_count > 0:
                    utilization_percent /= device_count
                    temperature_c /= device_count
            
            return GPUMetrics(
                device_count=device_count,
                devices=devices,
                total_memory_gb=total_memory_gb,
                used_memory_gb=used_memory_gb,
                free_memory_gb=free_memory_gb,
                utilization_percent=utilization_percent,
                temperature_c=temperature_c,
                power_usage_watts=power_usage_watts,
                power_limit_watts=power_limit_watts,
                memory_temperature_c=memory_temperature_c,
                is_mps_available=is_mps_available,
            )
        
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
            return GPUMetrics()
    
    def _collect_disk_metrics(self) -> DiskMetrics:
        """
        Collect disk metrics.
        
        Returns:
            DiskMetrics: Disk usage metrics
        """
        try:
            # Get disk usage for the current directory
            disk_usage = shutil.disk_usage(self.log_dir)
            total_gb = disk_usage.total / (1024 ** 3)
            used_gb = disk_usage.used / (1024 ** 3)
            free_gb = disk_usage.free / (1024 ** 3)
            percent_used = (disk_usage.used / disk_usage.total) * 100 if disk_usage.total > 0 else 0
            
            # Get disk I/O stats
            disk_io = psutil.disk_io_counters()
            read_count = disk_io.read_count if disk_io else 0
            write_count = disk_io.write_count if disk_io else 0
            read_bytes = disk_io.read_bytes if disk_io else 0
            write_bytes = disk_io.write_bytes if disk_io else 0
            read_time = disk_io.read_time if disk_io else 0
            write_time = disk_io.write_time if disk_io else 0
            
            # Calculate I/O rates if we have previous measurements
            if self.last_disk_metrics is not None and self.last_check_time is not None:
                time_diff = time.time() - self.last_check_time
                
                # Skip rate calculation if time difference is too small
                if time_diff > 0.1:
                    read_count_rate = (read_count - self.last_disk_metrics.read_count) / time_diff
                    write_count_rate = (write_count - self.last_disk_metrics.write_count) / time_diff
                    read_bytes_rate = (read_bytes - self.last_disk_metrics.read_bytes) / time_diff
                    write_bytes_rate = (write_bytes - self.last_disk_metrics.write_bytes) / time_diff
                
                # Log high I/O rates
                if time_diff > 0.1:
                    read_mb_per_sec = read_bytes_rate / (1024 * 1024)
                    write_mb_per_sec = write_bytes_rate / (1024 * 1024)
                    
                    if read_mb_per_sec > 100 or write_mb_per_sec > 100:
                        logger.debug(f"High disk I/O: Read {read_mb_per_sec:.1f} MB/s, Write {write_mb_per_sec:.1f} MB/s")
            
            metrics = DiskMetrics(
                total_gb=total_gb,
                used_gb=used_gb,
                free_gb=free_gb,
                percent_used=percent_used,
                read_count=read_count,
                write_count=write_count,
                read_bytes=read_bytes,
                write_bytes=write_bytes,
                read_time_ms=read_time,
                write_time_ms=write_time,
            )
            
            # Store for rate calculations
            self.last_disk_metrics = metrics
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
            return DiskMetrics()
    
    def _collect_network_metrics(self) -> NetworkMetrics:
        """
        Collect network metrics.
        
        Returns:
            NetworkMetrics: Network usage metrics
        """
        try:
            # Get network I/O stats
            net_io = psutil.net_io_counters()
            bytes_sent = net_io.bytes_sent
            bytes_recv = net_io.bytes_recv
            packets_sent = net_io.packets_sent
            packets_recv = net_io.packets_recv
            errin = net_io.errin
            errout = net_io.errout
            dropin = net_io.dropin
            dropout = net_io.dropout
            
            # Calculate bandwidth if we have previous measurements
            bandwidth_mbps = 0.0
            if self.last_network_metrics is not None and self.last_check_time is not None:
                time_diff = time.time() - self.last_check_time
                
                # Skip bandwidth calculation if time difference is too small
                if time_diff > 0.1:
                    bytes_sent_diff = bytes_sent - self.last_network_metrics.bytes_sent
                    bytes_recv_diff = bytes_recv - self.last_network_metrics.bytes_recv
                    total_bytes_diff = bytes_sent_diff + bytes_recv_diff
                    
                    # Convert to Mbps
                    bandwidth_mbps = (total_bytes_diff * 8) / (time_diff * 1024 * 1024)
            
            metrics = NetworkMetrics(
                bytes_sent=bytes_sent,
                bytes_recv=bytes_recv,
                packets_sent=packets_sent,
                packets_recv=packets_recv,
                errin=errin,
                errout=errout,
                dropin=dropin,
                dropout=dropout,
                bandwidth_mbps=bandwidth_mbps,
            )
            
            # Store for bandwidth calculation
            self.last_network_metrics = metrics
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
            return NetworkMetrics()
    
    def _collect_thermal_metrics(self) -> ThermalMetrics:
        """
        Collect thermal state metrics.
        
        Returns:
            ThermalMetrics: Thermal state metrics
        """
        try:
            # Initialize metrics
            cpu_temperature_c = 0.0
            gpu_temperature_c = 0.0
            is_throttling = False
            fan_speed_rpm = 0
            thermal_pressure = "nominal"
            
            # Get GPU temperature if available
            if self.monitor_gpu:
                gpu_metrics = self._get_latest_gpu_metrics()
                if gpu_metrics:
                    gpu_temperature_c = gpu_metrics.temperature_c
            
            # Platform-specific thermal monitoring
            if self.platform == SystemPlatform.MACOS or self.platform == SystemPlatform.MACOS_SILICON:
                # Try to get thermal info from macOS
                try:
                    # Check for thermal pressure on macOS 11+
                    if self.platform == SystemPlatform.MACOS_SILICON:
                        try:
                            # Use powermetrics to get thermal pressure
                            # This requires sudo, so it might not work
                            # result = subprocess.run(
                            #     ["sudo", "powermetrics", "-n", "1", "-i", "1000", "--show-thermal-pressure"],
                            #     capture_output=True, text=True, timeout=2
                            # )
                            # output = result.stdout
                            # if "thermal_pressure" in output:
                            #     if "CRITICAL" in output:
                            #         thermal_pressure = "critical"
                            #         is_throttling = True
                            #     elif "HEAVY" in output:
                            #         thermal_pressure = "heavy"
                            #         is_throttling = True
                            #     elif "MODERATE" in output:
                            #         thermal_pressure = "moderate"
                            
                            # Instead, use CPU usage pattern as a proxy for throttling
                            cpu_metrics = self._get_latest_cpu_metrics()
                            if cpu_metrics:
                                # Check for throttling pattern: high load with decreasing frequency
                                if cpu_metrics.usage_percent > 90 and cpu_metrics.frequency_mhz < 2000:
                                    is_throttling = True
                                    thermal_pressure = "heavy"
                        except Exception as e:
                            logger.debug(f"Error getting thermal pressure: {e}")
                    
                    # Try to get CPU temperature
                    # This requires additional tools that may not be available
                    # For now, use GPU temperature as a proxy
                    cpu_temperature_c = gpu_temperature_c
                
                except Exception as e:
                    logger.debug(f"Error getting macOS thermal info: {e}")
            
            elif self.platform == SystemPlatform.LINUX:
                # Try to get thermal info from Linux
                try:
                    # Check if sensors command is available
                    result = subprocess.run(
                        ["sensors"],
                        capture_output=True, text=True, timeout=2
                    )
                    output = result.stdout
                    
                    # Parse CPU temperature
                    cpu_temp_match = re.search(r"Core \d+:\s+\+(\d+\.\d+)°C", output)
                    if cpu_temp_match:
                        cpu_temperature_c = float(cpu_temp_match.group(1))
                    
                    # Check for throttling
                    if "ALARM" in output or "CRITICAL" in output:
                        is_throttling = True
                
                except Exception as e:
                    logger.debug(f"Error getting Linux thermal info: {e}")
            
            return ThermalMetrics(
                cpu_temperature_c=cpu_temperature_c,
                gpu_temperature_c=gpu_temperature_c,
                is_throttling=is_throttling,
                fan_speed_rpm=fan_speed_rpm,
                thermal_pressure=thermal_pressure,
            )
        
        except Exception as e:
            logger.error(f"Error collecting thermal metrics: {e}")
            return ThermalMetrics()
    
    def _analyze_system_health(self):
        """Analyze system health and efficiency."""
        try:
            # Initialize metrics
            health_score = 1.0
            cpu_efficiency = 1.0
            memory_efficiency = 1.0
            gpu_efficiency = 1.0
            io_efficiency = 1.0
            training_efficiency = 1.0
            bottleneck = "none"
            crash_probability = 0.0
            memory_leak_detected = False
            throttling_detected = False
            io_bottleneck_detected = False
            
            # Get latest metrics
            cpu_metrics = self._get_latest_cpu_metrics()
            memory_metrics = self._get_latest_memory_metrics()
            gpu_metrics = self._get_latest_gpu_metrics()
            disk_metrics = self._get_latest_disk_metrics()
            thermal_metrics = self._get_latest_thermal_metrics()
            
            # Check CPU health
            if cpu_metrics:
                # CPU efficiency decreases as usage gets very high
                if cpu_metrics.usage_percent > 95:
                    cpu_efficiency = 0.7
                    health_score -= 0.1
                    if bottleneck == "none":
                        bottleneck = "cpu"
                elif cpu_metrics.usage_percent > 85:
                    cpu_efficiency = 0.9
            
            # Check memory health
            if memory_metrics:
                # Memory efficiency decreases as usage gets very high
                if memory_metrics.percent_used > 95:
                    memory_efficiency = 0.5
                    health_score -= 0.2
                    crash_probability += 0.3
                    if bottleneck == "none":
                        bottleneck = "memory"
                elif memory_metrics.percent_used > 85:
                    memory_efficiency = 0.8
                    health_score -= 0.1
                    crash_probability += 0.1
                    if bottleneck == "none":
                        bottleneck = "memory"
                
                # Check for memory leaks
                if self.enable_memory_leak_detection and len(self.potential_memory_leaks) > 0:
                    memory_leak_detected = True
                    health_score -= 0.1
                    crash_probability += 0.1
            
            # Check GPU health
            if gpu_metrics and gpu_metrics.device_count > 0:
                # GPU efficiency decreases with high memory usage
                memory_utilization = gpu_metrics.used_memory_gb / gpu_metrics.total_memory_gb if gpu_metrics.total_memory_gb > 0 else 0
                if memory_utilization > 0.95:
                    gpu_efficiency = 0.6
                    health_score -= 0.15
                    crash_probability += 0.2
                    if bottleneck == "none":
                        bottleneck = "gpu"
                elif memory_utilization > 0.85:
                    gpu_efficiency = 0.8
                    health_score -= 0.05
                    if bottleneck == "none" and gpu_metrics.utilization_percent > 90:
                        bottleneck = "gpu"
                
                # Check temperature
                if gpu_metrics.temperature_c > 85:
                    gpu_efficiency *= 0.8
                    health_score -= 0.1
                    crash_probability += 0.1
            
            # Check thermal health
            if thermal_metrics:
                if thermal_metrics.is_throttling:
                    throttling_detected = True
                    health_score -= 0.2
                    cpu_efficiency *= 0.7
                    gpu_efficiency *= 0.7
                    training_efficiency *= 0.7
                    if bottleneck == "none":
                        bottleneck = "thermal"
            
            # Check disk health
            if disk_metrics:
                if disk_metrics.free_gb < 1.0:
                    io_efficiency = 0.5
                    health_score -= 0.2
                    crash_probability += 0.2
                    if bottleneck == "none":
                        bottleneck = "disk"
                elif disk_metrics.free_gb < 5.0:
                    io_efficiency = 0.8
                    health_score -= 0.05
                
                # Check for I/O bottlenecks
                if self.last_disk_metrics and self.last_check_time:
                    time_diff = time.time() - self.last_check_time
                    if time_diff > 0.1:
                        read_bytes_rate = (disk_metrics.read_bytes - self.last_disk_metrics.read_bytes) / time_diff
                        write_bytes_rate = (disk_metrics.write_bytes - self.last_disk_metrics.write_bytes) / time_diff
                        
                        # High I/O rates can indicate bottlenecks
                        if read_bytes_rate > 100 * 1024 * 1024 or write_bytes_rate > 100 * 1024 * 1024:
                            io_bottleneck_detected = True
                            io_efficiency *= 0.8
                            if bottleneck == "none":
                                bottleneck = "disk"
            
            # Check training efficiency
            if len(self.training_metrics_history) > 1:
                # Calculate training throughput trend
                recent_metrics = self.training_metrics_history[-5:]
                if len(recent_metrics) > 1:
                    throughputs = [m.samples_per_second for m in recent_metrics if m.samples_per_second > 0]
                    if throughputs:
                        avg_throughput = sum(throughputs) / len(throughputs)
                        if avg_throughput < 10:  # Very low throughput
                            training_efficiency *= 0.7
                            if bottleneck == "none":
                                bottleneck = "training"
            
            # Ensure health score is between 0 and 1
            health_score = max(0.0, min(1.0, health_score))
            crash_probability = max(0.0, min(0.9, crash_probability))
            
            # Create system health metrics
            health_metrics = SystemHealthMetrics(
                health_score=health_score,
                cpu_efficiency=cpu_efficiency,
                memory_efficiency=memory_efficiency,
                gpu_efficiency=gpu_efficiency,
                io_efficiency=io_efficiency,
                training_efficiency=training_efficiency,
                bottleneck=bottleneck,
                crash_probability=crash_probability,
                memory_leak_detected=memory_leak_detected,
                throttling_detected=throttling_detected,
                io_bottleneck_detected=io_bottleneck_detected,
            )
            
            # Add to history
            self._add_to_history(self.system_health_history, health_metrics)
            
            # Log metrics
            self._log_metrics(health_metrics)
            
            # Generate alerts based on health
            self._generate_health_alerts(health_metrics)
        
        except Exception as e:
            logger.error(f"Error analyzing system health: {e}")
    
    def _check_memory_leaks(self):
        """Check for potential memory leaks."""
        if not self.memory_metrics_history or len(self.memory_metrics_history) < 10:
            return
        
        try:
            # Get memory usage over time
            timestamps = [m.timestamp - self.start_time for m in self.memory_metrics_history]
            memory_usage = [m.used_gb for m in self.memory_metrics_history]
            
            # Skip if we don't have enough data points
            if len(timestamps) < 10:
                return
            
            # Simple linear regression to detect trend
            if len(timestamps) >= 2:
                slope, intercept = np.polyfit(timestamps, memory_usage, 1)
                
                # Convert to MB/hour for easier interpretation
                growth_rate_mb_per_hour = slope * 3600 * 1000  # Convert to MB/hour
                
                # Check if growth rate is significant
                if growth_rate_mb_per_hour > 500:  # More than 500 MB/hour
                    # Check if we've already detected this leak
                    leak_info = {
                        "detected_at": time.time(),
                        "growth_rate_mb_per_hour": growth_rate_mb_per_hour,
                        "initial_memory_gb": self.memory_baseline.used_gb if self.memory_baseline else 0.0,
                        "current_memory_gb": self.memory_metrics_history[-1].used_gb,
                    }
                    
                    # Only add if we don't already have a similar leak
                    if not self.potential_memory_leaks or time.time() - self.potential_memory_leaks[-1]["detected_at"] > 300:
                        self.potential_memory_leaks.append(leak_info)
                        
                        # Generate alert
                        self._generate_alert(
                            resource_type=ResourceType.MEMORY,
                            level=AlertLevel.WARNING,
                            message=f"Potential memory leak detected: {growth_rate_mb_per_hour:.1f} MB/hour",
                            details=leak_info,
                            suggested_actions=[
                                "Check for tensor accumulation in training loop",
                                "Ensure proper cleanup of large objects",
                                "Consider using torch.no_grad() for inference",
                                "Run garbage collection more frequently",
                            ]
                        )
                        
                        # Try to recover some memory
                        if self.model is not None:
                            # Clear PyTorch caches
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # Run garbage collection
                            gc.collect()
        
        except Exception as e:
            logger.error(f"Error checking for memory leaks: {e}")
    
    def _update_adaptive_recommendations(self):
        """Update adaptive resource management recommendations."""
        if not self.enable_adaptive_resources:
            return
        
        try:
            # Get latest metrics
            memory_metrics = self._get_latest_memory_metrics()
            gpu_metrics = self._get_latest_gpu_metrics()
            thermal_metrics = self._get_latest_thermal_metrics()
            system_health = self._get_latest_system_health()
            
            # Get current batch size from training metrics
            current_batch_size = None
            if self.training_metrics_history:
                current_batch_size = self.training_metrics_history[-1].batch_size
            
            if current_batch_size is None or current_batch_size <= 0:
                return
            
            # Initialize recommended batch size to current
            recommended_batch_size = current_batch_size
            
            # Adjust based on memory pressure
            if memory_metrics and memory_metrics.percent_used > 90:
                # Reduce batch size by 25%
                recommended_batch_size = max(1, int(current_batch_size * 0.75))
            elif memory_metrics and memory_metrics.percent_used < 50:
                # Increase batch size by 20%
                recommended_batch_size = int(current_batch_size * 1.2)
            
            # Adjust based on GPU memory
            if gpu_metrics and gpu_metrics.device_count > 0:
                memory_utilization = gpu_metrics.used_memory_gb / gpu_metrics.total_memory_gb if gpu_metrics.total_memory_gb > 0 else 0
                if memory_utilization > 0.9:
                    # Reduce batch size by 25%
                    recommended_batch_size = min(recommended_batch_size, max(1, int(current_batch_size * 0.75)))
                elif memory_utilization < 0.5 and gpu_metrics.utilization_percent > 80:
                    # Increase batch size by 20%
                    recommended_batch_size = max(recommended_batch_size, int(current_batch_size * 1.2))
            
            # Adjust based on thermal state
            if thermal_metrics and thermal_metrics.is_throttling:
                # Reduce batch size by 50% if throttling
                recommended_batch_size = min(recommended_batch_size, max(1, int(current_batch_size * 0.5)))
            
            # Adjust based on system health
            if system_health and system_health.health_score < 0.7:
                # Reduce batch size by 25% if system health is poor
                recommended_batch_size = min(recommended_batch_size, max(1, int(current_batch_size * 0.75)))
            
            # Only update if recommendation is different
            if recommended_batch_size != current_batch_size and recommended_batch_size != self.recommended_batch_size:
                self.recommended_batch_size = recommended_batch_size
                
                # Log recommendation
                logger.info(f"Recommended batch size: {recommended_batch_size} (current: {current_batch_size})")
                
                # Generate alert
                if recommended_batch_size < current_batch_size:
                    self._generate_alert(
                        resource_type=ResourceType.SYSTEM,
                        level=AlertLevel.INFO,
                        message=f"Recommended reducing batch size from {current_batch_size} to {recommended_batch_size}",
                        details={
                            "current_batch_size": current_batch_size,
                            "recommended_batch_size": recommended_batch_size,
                            "memory_percent_used": memory_metrics.percent_used if memory_metrics else None,
                            "gpu_memory_utilization": memory_utilization if gpu_metrics and gpu_metrics.device_count > 0 else None,
                            "is_throttling": thermal_metrics.is_throttling if thermal_metrics else None,
                            "health_score": system_health.health_score if system_health else None,
                        },
                        suggested_actions=[
                            f"Reduce batch size to {recommended_batch_size}",
                            "Consider gradient accumulation to maintain effective batch size",
                        ]
                    )
        
        except Exception as e:
            logger.error(f"Error updating adaptive recommendations: {e}")
    
    def _check_training_health(self, metrics: TrainingMetrics):
        """
        Check for training health issues.
        
        Args:
            metrics: Training metrics
        """
        try:
            # Check for NaN or inf loss
            if np.isnan(metrics.loss) or np.isinf(metrics.loss):
                self._generate_alert(
                    resource_type=ResourceType.TRAINING,
                    level=AlertLevel.CRITICAL,
                    message=f"Training diverged: loss is {metrics.loss}",
                    details={
                        "step": metrics.step,
                        "epoch": metrics.epoch,
                        "loss": metrics.loss,
                        "learning_rate": metrics.learning_rate,
                    },
                    suggested_actions=[
                        "Reduce learning rate",
                        "Check for numerical instability in model",
                        "Inspect input data for anomalies",
                        "Add gradient clipping",
                    ]
                )
            
            # Check for unusually high loss
            if metrics.loss > 1000:
                self._generate_alert(
                    resource_type=ResourceType.TRAINING,
                    level=AlertLevel.WARNING,
                    message=f"Unusually high loss: {metrics.loss:.2f}",
                    details={
                        "step": metrics.step,
                        "epoch": metrics.epoch,
                        "loss": metrics.loss,
                        "learning_rate": metrics.learning_rate,
                    },
                    suggested_actions=[
                        "Reduce learning rate",
                        "Check for data preprocessing issues",
                        "Inspect model initialization",
                    ]
                )
            
            # Check for very low throughput
            if metrics.samples_per_second > 0 and metrics.samples_per_second < 1:
                self._generate_alert(
                    resource_type=ResourceType.TRAINING,
                    level=AlertLevel.WARNING,
                    message=f"Very low throughput: {metrics.samples_per_second:.2f} samples/second",
                    details={
                        "step": metrics.step,
                        "epoch": metrics.epoch,
                        "samples_per_second": metrics.samples_per_second,
                        "batch_size": metrics.batch_size,
                    },
                    suggested_actions=[
                        "Check for bottlenecks in data loading",
                        "Reduce model complexity",
                        "Use mixed precision training",
                        "Check for CPU-GPU synchronization issues",
                    ]
                )
            
            # Check for high gradient norm
            if metrics.grad_norm > 100:
                self._generate_alert(
                    resource_type=ResourceType.TRAINING,
                    level=AlertLevel.WARNING,
                    message=f"High gradient norm: {metrics.grad_norm:.2f}",
                    details={
                        "step": metrics.step,
                        "epoch": metrics.epoch,
                        "grad_norm": metrics.grad_norm,
                        "learning_rate": metrics.learning_rate,
                    },
                    suggested_actions=[
                        