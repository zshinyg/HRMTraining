#!/usr/bin/env python
"""
Reliability Monitor for HRM Training

A comprehensive system monitoring tool for HRM vs Transformer hypothesis validation.
Provides real-time monitoring, anomaly detection, and recovery recommendations
with special support for Apple Silicon (M1/M2/M3) hardware.

Usage:
    python scripts/reliability_monitor.py --mode [train|monitor|benchmark]
    python scripts/reliability_monitor.py --config configs/m1_optimized_training.yaml
    python scripts/reliability_monitor.py --attach <pid> --alert-threshold 0.9

Author: @zshinyg
Date: 2025-08-05
"""

import argparse
import datetime
import json
import logging
import os
import platform
import psutil
import signal
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/reliability_monitor.log"),
    ],
)
logger = logging.getLogger("reliability_monitor")

# Constants
DEFAULT_CONFIG_PATH = "configs/m1_optimized_training.yaml"
HEARTBEAT_PATH = "logs/heartbeat.txt"
ALERT_LOG_PATH = "logs/alerts.log"
BENCHMARK_RESULTS_PATH = "logs/benchmark_results.json"
RECOVERY_SCRIPT_PATH = "scripts/recover.sh"
MPS_MEMORY_THRESHOLD = 0.9  # 90% of available memory
CPU_THRESHOLD = 0.9  # 90% CPU usage
DISK_THRESHOLD = 0.9  # 90% disk usage
LOSS_SPIKE_FACTOR = 2.0  # Alert if loss increases by this factor
GRAD_NORM_THRESHOLD = 10.0  # Alert if gradient norm exceeds this value
HEARTBEAT_INTERVAL = 60  # seconds
THROUGHPUT_WINDOW = 50  # steps


class AlertLevel(Enum):
    """Alert levels for the monitoring system."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class SystemMetrics:
    """Container for system metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    mps_memory_used: Optional[float] = None
    mps_memory_total: Optional[float] = None
    mps_utilization: Optional[float] = None
    temperature: Optional[Dict[str, float]] = None
    network_io: Optional[Dict[str, int]] = None
    battery_percent: Optional[float] = None
    power_plugged: Optional[bool] = None


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    timestamp: float = field(default_factory=time.time)
    step: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    throughput: Optional[float] = None  # samples/second
    batch_time: Optional[float] = None  # seconds
    memory_allocated: Optional[float] = None
    memory_reserved: Optional[float] = None
    forward_time: Optional[float] = None
    backward_time: Optional[float] = None
    optimizer_time: Optional[float] = None


class ReliabilityMonitor:
    """
    Comprehensive system monitoring tool for HRM training.
    
    Features:
    - System resource monitoring (CPU, memory, disk, MPS)
    - Training metrics tracking (loss, gradients, throughput)
    - Anomaly detection and alert generation
    - Heartbeat logging for external monitoring
    - Performance benchmarking
    - Recovery recommendations
    """
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, mode: str = "monitor"):
        """
        Initialize the reliability monitor.
        
        Args:
            config_path: Path to the configuration file
            mode: Monitoring mode (train, monitor, benchmark)
        """
        self.config_path = config_path
        self.mode = mode
        self.config = self._load_config()
        self.system_metrics_history: List[SystemMetrics] = []
        self.training_metrics_history: List[TrainingMetrics] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.is_running = False
        self.monitoring_thread = None
        self.attached_pid = None
        self.last_heartbeat = time.time()
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        
        # Initialize metrics
        self.current_system_metrics = SystemMetrics()
        self.current_training_metrics = TrainingMetrics()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        logger.info(f"Reliability Monitor initialized in {mode} mode")
        logger.info(f"Using configuration from {config_path}")
        logger.info(f"System: {platform.system()} {platform.release()} on {platform.machine()}")
        logger.info(f"Python: {platform.python_version()}, PyTorch: {torch.__version__}")
        
        # Check if running on Apple Silicon
        self.is_apple_silicon = platform.machine() == "arm64" and platform.system() == "Darwin"
        if self.is_apple_silicon:
            logger.info("Detected Apple Silicon hardware, enabling MPS monitoring")
            if not torch.backends.mps.is_available():
                logger.warning("MPS is not available despite running on Apple Silicon")
        
        # Initialize MPS environment variables if needed
        if self.is_apple_silicon and mode in ["train", "benchmark"]:
            self._setup_mps_environment()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {self.config_path}: {e}")
            # Return default configuration
            return {
                "system": {"device": "mps" if torch.backends.mps.is_available() else "cpu"},
                "monitoring": {
                    "heartbeat": {"enabled": True, "interval_seconds": HEARTBEAT_INTERVAL},
                    "wandb": {"enabled": False},
                },
                "safety": {
                    "detect_anomaly": True,
                    "loss_watchdog": {"enabled": True, "threshold": 50.0},
                    "auto_restart": {"enabled": True, "max_retries": 3},
                },
                "memory": {"optimize_memory_usage": True, "empty_cache_freq": 100},
            }
    
    def _setup_mps_environment(self):
        """Set up environment variables for optimal MPS performance."""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.9"
        logger.info("Set MPS environment variables for optimal performance")
    
    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.stop()
        sys.exit(0)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system metrics."""
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=1) / 100.0,
            memory_percent=psutil.virtual_memory().percent / 100.0,
            disk_percent=psutil.disk_usage("/").percent / 100.0,
        )
        
        # Collect battery information if available
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
            if battery:
                metrics.battery_percent = battery.percent / 100.0
                metrics.power_plugged = battery.power_plugged
        
        # Collect network I/O
        net_io = psutil.net_io_counters()
        metrics.network_io = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
        }
        
        # Collect MPS metrics if on Apple Silicon
        if self.is_apple_silicon:
            try:
                if torch.backends.mps.is_available():
                    metrics.mps_memory_used = torch.mps.current_allocated_memory() / (1024 ** 3)  # GB
                    metrics.mps_memory_total = torch.mps.driver_allocated_memory() / (1024 ** 3)  # GB
                    
                    # Get GPU utilization using powermetrics (requires sudo)
                    # This is commented out as it requires sudo privileges
                    # try:
                    #     result = subprocess.run(
                    #         ["sudo", "powermetrics", "-n", "1", "-i", "1000", "--samplers", "gpu_power"],
                    #         capture_output=True, text=True, timeout=2
                    #     )
                    #     if result.returncode == 0:
                    #         # Parse powermetrics output to get GPU utilization
                    #         for line in result.stdout.split("\n"):
                    #             if "GPU active residency" in line:
                    #                 metrics.mps_utilization = float(line.split("%")[0].strip().split()[-1]) / 100.0
                    # except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                    #     pass
            except Exception as e:
                logger.warning(f"Failed to collect MPS metrics: {e}")
        
        return metrics
    
    def _collect_training_metrics(self, model=None, optimizer=None, loss=None) -> TrainingMetrics:
        """
        Collect training metrics.
        
        Args:
            model: PyTorch model (optional)
            optimizer: PyTorch optimizer (optional)
            loss: Current loss value (optional)
        
        Returns:
            TrainingMetrics object
        """
        metrics = TrainingMetrics(timestamp=time.time())
        
        if loss is not None:
            metrics.loss = float(loss)
        
        if model is not None:
            # Collect memory metrics
            if torch.cuda.is_available():
                metrics.memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                metrics.memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
            elif torch.backends.mps.is_available():
                metrics.memory_allocated = torch.mps.current_allocated_memory() / (1024 ** 3)  # GB
                metrics.memory_reserved = torch.mps.driver_allocated_memory() / (1024 ** 3)  # GB
            
            # Collect gradient norm if available
            if all(p.grad is not None for p in model.parameters() if p.requires_grad):
                grad_norm = torch.norm(
                    torch.stack([
                        torch.norm(p.grad.detach()) 
                        for p in model.parameters() 
                        if p.grad is not None
                    ])
                ).item()
                metrics.grad_norm = grad_norm
        
        if optimizer is not None:
            # Get learning rate from optimizer
            for param_group in optimizer.param_groups:
                if "lr" in param_group:
                    metrics.learning_rate = param_group["lr"]
                    break
        
        return metrics
    
    def _update_heartbeat(self):
        """Update heartbeat file for external monitoring."""
        if self.config.get("monitoring", {}).get("heartbeat", {}).get("enabled", True):
            try:
                heartbeat_data = {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "mode": self.mode,
                    "system_metrics": {
                        "cpu_percent": self.current_system_metrics.cpu_percent,
                        "memory_percent": self.current_system_metrics.memory_percent,
                        "disk_percent": self.current_system_metrics.disk_percent,
                    },
                    "training_metrics": {
                        "step": self.current_training_metrics.step,
                        "loss": self.current_training_metrics.loss,
                    } if self.current_training_metrics.loss is not None else None,
                    "status": "healthy",
                }
                
                with open(HEARTBEAT_PATH, "w") as f:
                    json.dump(heartbeat_data, f)
                
                self.last_heartbeat = time.time()
                logger.debug("Updated heartbeat file")
            except Exception as e:
                logger.error(f"Failed to update heartbeat: {e}")
    
    def _check_for_anomalies(self) -> List[Dict[str, Any]]:
        """
        Check for anomalies in system and training metrics.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Check system metrics
        if self.current_system_metrics.cpu_percent > CPU_THRESHOLD:
            alerts.append({
                "timestamp": time.time(),
                "level": AlertLevel.WARNING.value,
                "message": f"High CPU usage: {self.current_system_metrics.cpu_percent:.1%}",
                "metric": "cpu_percent",
                "value": self.current_system_metrics.cpu_percent,
                "threshold": CPU_THRESHOLD,
                "recommendation": "Consider reducing batch size or number of workers",
            })
        
        if self.current_system_metrics.memory_percent > DISK_THRESHOLD:
            alerts.append({
                "timestamp": time.time(),
                "level": AlertLevel.WARNING.value,
                "message": f"High memory usage: {self.current_system_metrics.memory_percent:.1%}",
                "metric": "memory_percent",
                "value": self.current_system_metrics.memory_percent,
                "threshold": DISK_THRESHOLD,
                "recommendation": "Reduce batch size or enable gradient checkpointing",
            })
        
        if self.current_system_metrics.disk_percent > DISK_THRESHOLD:
            alerts.append({
                "timestamp": time.time(),
                "level": AlertLevel.WARNING.value,
                "message": f"High disk usage: {self.current_system_metrics.disk_percent:.1%}",
                "metric": "disk_percent",
                "value": self.current_system_metrics.disk_percent,
                "threshold": DISK_THRESHOLD,
                "recommendation": "Clean up old checkpoints or logs",
            })
        
        # Check MPS metrics if available
        if (self.current_system_metrics.mps_memory_used is not None and 
            self.current_system_metrics.mps_memory_total is not None):
            mps_usage = self.current_system_metrics.mps_memory_used / self.current_system_metrics.mps_memory_total
            if mps_usage > MPS_MEMORY_THRESHOLD:
                alerts.append({
                    "timestamp": time.time(),
                    "level": AlertLevel.WARNING.value,
                    "message": f"High MPS memory usage: {mps_usage:.1%}",
                    "metric": "mps_memory_usage",
                    "value": mps_usage,
                    "threshold": MPS_MEMORY_THRESHOLD,
                    "recommendation": "Reduce batch size, enable gradient checkpointing, or use torch.compile",
                })
        
        # Check training metrics if available
        if len(self.training_metrics_history) >= 2:
            # Check for loss spikes
            if (self.current_training_metrics.loss is not None and 
                self.training_metrics_history[-2].loss is not None):
                loss_ratio = self.current_training_metrics.loss / max(self.training_metrics_history[-2].loss, 1e-8)
                if loss_ratio > LOSS_SPIKE_FACTOR:
                    alerts.append({
                        "timestamp": time.time(),
                        "level": AlertLevel.ERROR.value,
                        "message": f"Loss spike detected: {loss_ratio:.2f}x increase",
                        "metric": "loss_ratio",
                        "value": loss_ratio,
                        "threshold": LOSS_SPIKE_FACTOR,
                        "recommendation": "Check for NaN/Inf values, reduce learning rate, or restore from previous checkpoint",
                    })
            
            # Check for NaN/Inf loss
            if self.current_training_metrics.loss is not None:
                if not np.isfinite(self.current_training_metrics.loss):
                    alerts.append({
                        "timestamp": time.time(),
                        "level": AlertLevel.CRITICAL.value,
                        "message": "Non-finite loss detected",
                        "metric": "loss",
                        "value": self.current_training_metrics.loss,
                        "recommendation": "Restore from previous checkpoint, reduce learning rate, and enable gradient clipping",
                    })
            
            # Check for high gradient norm
            if self.current_training_metrics.grad_norm is not None:
                if self.current_training_metrics.grad_norm > GRAD_NORM_THRESHOLD:
                    alerts.append({
                        "timestamp": time.time(),
                        "level": AlertLevel.WARNING.value,
                        "message": f"High gradient norm: {self.current_training_metrics.grad_norm:.2f}",
                        "metric": "grad_norm",
                        "value": self.current_training_metrics.grad_norm,
                        "threshold": GRAD_NORM_THRESHOLD,
                        "recommendation": "Enable gradient clipping or reduce learning rate",
                    })
            
            # Check for decreasing throughput
            if (len(self.training_metrics_history) > THROUGHPUT_WINDOW and
                all(m.throughput is not None for m in self.training_metrics_history[-THROUGHPUT_WINDOW:])):
                recent_throughput = self.current_training_metrics.throughput
                past_throughput = np.mean([
                    m.throughput for m in self.training_metrics_history[-THROUGHPUT_WINDOW:-THROUGHPUT_WINDOW//2]
                ])
                if recent_throughput < 0.7 * past_throughput:
                    alerts.append({
                        "timestamp": time.time(),
                        "level": AlertLevel.WARNING.value,
                        "message": f"Throughput degradation: {recent_throughput:.2f} vs {past_throughput:.2f} samples/sec",
                        "metric": "throughput_ratio",
                        "value": recent_throughput / past_throughput,
                        "threshold": 0.7,
                        "recommendation": "Check for background processes, thermal throttling, or memory leaks",
                    })
        
        # Check heartbeat
        heartbeat_interval = self.config.get("monitoring", {}).get("heartbeat", {}).get("interval_seconds", HEARTBEAT_INTERVAL)
        if time.time() - self.last_heartbeat > 2 * heartbeat_interval:
            alerts.append({
                "timestamp": time.time(),
                "level": AlertLevel.WARNING.value,
                "message": f"Heartbeat missing for {time.time() - self.last_heartbeat:.1f} seconds",
                "metric": "heartbeat_delay",
                "value": time.time() - self.last_heartbeat,
                "threshold": 2 * heartbeat_interval,
                "recommendation": "Check for process hang or high system load",
            })
        
        return alerts
    
    def _log_alerts(self, alerts: List[Dict[str, Any]]):
        """Log alerts and take appropriate actions."""
        for alert in alerts:
            level = alert["level"]
            message = alert["message"]
            
            # Log the alert
            if level == AlertLevel.INFO.value:
                logger.info(f"ALERT: {message}")
            elif level == AlertLevel.WARNING.value:
                logger.warning(f"ALERT: {message}")
            elif level == AlertLevel.ERROR.value:
                logger.error(f"ALERT: {message}")
            elif level == AlertLevel.CRITICAL.value:
                logger.critical(f"ALERT: {message}")
            
            # Add to alert history
            self.alert_history.append(alert)
            
            # Write to alert log
            try:
                with open(ALERT_LOG_PATH, "a") as f:
                    f.write(json.dumps(alert) + "\n")
            except Exception as e:
                logger.error(f"Failed to write to alert log: {e}")
            
            # Take action for critical alerts
            if level == AlertLevel.CRITICAL.value:
                if self.config.get("safety", {}).get("auto_restart", {}).get("enabled", True):
                    logger.critical("Critical alert triggered auto-restart")
                    self._trigger_recovery()
    
    def _trigger_recovery(self):
        """Trigger recovery script for critical issues."""
        if os.path.exists(RECOVERY_SCRIPT_PATH):
            try:
                logger.info(f"Triggering recovery script: {RECOVERY_SCRIPT_PATH}")
                subprocess.Popen(["bash", RECOVERY_SCRIPT_PATH], start_new_session=True)
            except Exception as e:
                logger.error(f"Failed to trigger recovery script: {e}")
        else:
            logger.error(f"Recovery script not found: {RECOVERY_SCRIPT_PATH}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect system metrics
                self.current_system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(self.current_system_metrics)
                
                # Check for anomalies
                alerts = self._check_for_anomalies()
                if alerts:
                    self._log_alerts(alerts)
                
                # Update heartbeat
                self._update_heartbeat()
                
                # Trim history to prevent memory bloat
                max_history = 1000
                if len(self.system_metrics_history) > max_history:
                    self.system_metrics_history = self.system_metrics_history[-max_history:]
                if len(self.training_metrics_history) > max_history:
                    self.training_metrics_history = self.training_metrics_history[-max_history:]
                if len(self.alert_history) > max_history:
                    self.alert_history = self.alert_history[-max_history:]
                
                # If attached to a process, check if it's still running
                if self.attached_pid is not None:
                    if not psutil.pid_exists(self.attached_pid):
                        logger.warning(f"Attached process {self.attached_pid} no longer exists")
                        self.stop()
                        break
                
                # Sleep for a short interval
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Sleep longer on error
    
    def start(self):
        """Start the monitoring system."""
        if self.is_running:
            logger.warning("Monitoring is already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Monitoring started")
    
    def stop(self):
        """Stop the monitoring system."""
        if not self.is_running:
            logger.warning("Monitoring is not running")
            return
        
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Monitoring stopped")
    
    def attach(self, pid: int):
        """
        Attach to an existing process for monitoring.
        
        Args:
            pid: Process ID to monitor
        """
        if not psutil.pid_exists(pid):
            logger.error(f"Process {pid} does not exist")
            return False
        
        self.attached_pid = pid
        logger.info(f"Attached to process {pid}")
        return True
    
    def log_training_metrics(self, step: int, loss: float, model=None, optimizer=None, 
                            batch_time: float = None, samples: int = None):
        """
        Log training metrics.
        
        Args:
            step: Current training step
            loss: Current loss value
            model: PyTorch model (optional)
            optimizer: PyTorch optimizer (optional)
            batch_time: Time taken for the current batch (optional)
            samples: Number of samples in the current batch (optional)
        """
        metrics = self._collect_training_metrics(model, optimizer, loss)
        metrics.step = step
        metrics.batch_time = batch_time
        
        # Calculate throughput if batch time and samples are provided
        if batch_time is not None and samples is not None and batch_time > 0:
            metrics.throughput = samples / batch_time
        
        self.current_training_metrics = metrics
        self.training_metrics_history.append(metrics)
        
        # Check for anomalies
        alerts = self._check_for_anomalies()
        if alerts:
            self._log_alerts(alerts)
        
        # Update heartbeat
        self._update_heartbeat()
    
    def benchmark(self, model, batch_size: int = 4, seq_len: int = 256, 
                 num_batches: int = 10, warmup: int = 2):
        """
        Run a performance benchmark.
        
        Args:
            model: PyTorch model to benchmark
            batch_size: Batch size for the benchmark
            seq_len: Sequence length for the benchmark
            num_batches: Number of batches to run
            warmup: Number of warmup batches
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Starting benchmark with batch_size={batch_size}, seq_len={seq_len}")
        
        # Determine device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Move model to device
        model.to(device)
        model.eval()
        
        # Create dummy inputs
        vocab_size = getattr(model.config, "vocab_size", 50257)
        input_shape = (batch_size, seq_len)
        
        # Timing results
        forward_times = []
        memory_usage = []
        
        # Run benchmark
        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                input_ids = torch.randint(0, vocab_size, input_shape, device=device)
                attention_mask = torch.ones_like(input_ids)
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Benchmark
            for i in range(num_batches):
                # Generate random inputs
                input_ids = torch.randint(0, vocab_size, input_shape, device=device)
                attention_mask = torch.ones_like(input_ids)
                
                # Clear cache
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
                
                # Measure memory before
                if device.type == "cuda":
                    mem_before = torch.cuda.memory_allocated()
                elif device.type == "mps":
                    mem_before = torch.mps.current_allocated_memory()
                else:
                    mem_before = 0
                
                # Time forward pass
                start_time = time.time()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    # No explicit synchronization for MPS
                    pass
                
                end_time = time.time()
                
                # Measure memory after
                if device.type == "cuda":
                    mem_after = torch.cuda.memory_allocated()
                elif device.type == "mps":
                    mem_after = torch.mps.current_allocated_memory()
                else:
                    mem_after = 0
                
                # Record results
                forward_times.append(end_time - start_time)
                memory_usage.append(mem_after - mem_before)
                
                logger.info(f"Batch {i+1}/{num_batches}: {forward_times[-1]:.4f} sec")
        
        # Calculate statistics
        forward_times = np.array(forward_times)
        memory_usage = np.array(memory_usage)
        
        results = {
            "timestamp": time.time(),
            "device": str(device),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_batches": num_batches,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "forward_time": {
                "mean": float(np.mean(forward_times)),
                "std": float(np.std(forward_times)),
                "min": float(np.min(forward_times)),
                "max": float(np.max(forward_times)),
                "p50": float(np.percentile(forward_times, 50)),
                "p90": float(np.percentile(forward_times, 90)),
                "p95": float(np.percentile(forward_times, 95)),
            },
            "throughput": {
                "samples_per_second": float(batch_size / np.mean(forward_times)),
                "tokens_per_second": float(batch_size * seq_len / np.mean(forward_times)),
            },
            "memory": {
                "mean_bytes": float(np.mean(memory_usage)),
                "peak_bytes": float(np.max(memory_usage)),
                "bytes_per_token": float(np.mean(memory_usage) / (batch_size * seq_len)),
            },
        }
        
        # Save results
        try:
            with open(BENCHMARK_RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Benchmark results saved to {BENCHMARK_RESULTS_PATH}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
        
        # Log summary
        logger.info(f"Benchmark complete:")
        logger.info(f"  Mean forward time: {results['forward_time']['mean']:.4f} sec")
        logger.info(f"  Throughput: {results['throughput']['samples_per_second']:.2f} samples/sec")
        logger.info(f"  Throughput: {results['throughput']['tokens_per_second']:.2f} tokens/sec")
        logger.info(f"  Memory per token: {results['memory']['bytes_per_token'] / 1024:.2f} KB")
        
        return results
    
    def system_optimization_check(self) -> Dict[str, Any]:
        """
        Run a system optimization check and provide recommendations.
        
        Returns:
            Dictionary with check results and recommendations
        """
        logger.info("Running system optimization check")
        
        results = {
            "timestamp": time.time(),
            "system": {
                "os": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "cpu_count": os.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024 ** 3),
                "disk_free_gb": psutil.disk_usage("/").free / (1024 ** 3),
            },
            "checks": {},
            "recommendations": [],
        }
        
        # Check PyTorch installation
        results["checks"]["pytorch"] = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        }
        
        if self.is_apple_silicon and not torch.backends.mps.is_available():
            results["recommendations"].append({
                "priority": "high",
                "message": "MPS is not available despite running on Apple Silicon",
                "action": "Reinstall PyTorch with MPS support: pip install torch torchvision",
            })
        
        # Check environment variables
        results["checks"]["env_vars"] = {
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        }
        
        if self.is_apple_silicon and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            results["recommendations"].append({
                "priority": "medium",
                "message": "PYTORCH_ENABLE_MPS_FALLBACK is not set",
                "action": "Set PYTORCH_ENABLE_MPS_FALLBACK=1 to avoid crashes on unsupported ops",
            })
        
        # Check disk space
        disk_free_gb = psutil.disk_usage("/").free / (1024 ** 3)
        if disk_free_gb < 10:
            results["recommendations"].append({
                "priority": "high",
                "message": f"Low disk space: {disk_free_gb:.1f} GB free",
                "action": "Free up disk space or clean up old checkpoints",
            })
        
        # Check for common optimization flags in config
        if self.config:
            results["checks"]["config"] = {
                "gradient_checkpointing": self.config.get("training", {}).get("gradient_checkpointing", False),
                "gradient_accumulation": self.config.get("training", {}).get("gradient_accumulation_steps", 1),
                "mixed_precision": self.config.get("system", {}).get("precision", None),
                "compile_mode": self.config.get("memory", {}).get("compile_mode", None),
            }
            
            if not self.config.get("training", {}).get("gradient_checkpointing", False):
                results["recommendations"].append({
                    "priority": "medium",
                    "message": "Gradient checkpointing is not enabled",
                    "action": "Enable gradient_checkpointing in config for memory savings",
                })
            
            if self.config.get("training", {}).get("gradient_accumulation_steps", 1) < 4 and self.is_apple_silicon:
                results["recommendations"].append({
                    "priority": "medium",
                    "message": "Low gradient accumulation steps for M1",
                    "action": "Increase gradient_accumulation_steps to 4-8 for better M1 performance",
                })
            
            if not self.config.get("memory", {}).get("use_compile", False) and self.is_apple_silicon:
                results["recommendations"].append({
                    "priority": "medium",
                    "message": "torch.compile is not enabled",
                    "action": "Enable use_compile: true in memory section for better performance",
                })
        
        # Check for optimal batch size
        if self.is_apple_silicon:
            # Estimate optimal batch size based on model size and available memory
            model_size_gb = 0.1  # Default estimate
            if "model" in self.config:
                hidden_size = self.config["model"].get("hidden_size", 768)
                num_layers = (
                    self.config["model"].get("high_level_layers", 2) + 
                    self.config["model"].get("low_level_layers", 4)
                )
                vocab_size = self.config["model"].get("vocab_size", 50257)
                
                # Rough estimate of model size in parameters
                params = (
                    hidden_size * hidden_size * num_layers * 4 +  # Layers
                    vocab_size * hidden_size +  # Embeddings
                    hidden_size * vocab_size  # Output projection
                )
                
                # Convert to GB (4 bytes per parameter)
                model_size_gb = params * 4 / (1024 ** 3)
            
            # Get available memory
            mem_gb = psutil.virtual_memory().total / (1024 ** 3)
            
            # Estimate optimal batch size
            optimal_batch = max(1, int((mem_gb * 0.5) / model_size_gb))
            current_batch = self.config.get("data", {}).get("batch_size", 8)
            
            if current_batch > optimal_batch:
                results["recommendations"].append({
                    "priority": "high",
                    "message": f"Batch size {current_batch} may be too large for available memory",
                    "action": f"Reduce batch size to {optimal_batch} or enable gradient checkpointing",
                })
            elif current_batch < optimal_batch * 0.5:
                results["recommendations"].append({
                    "priority": "low",
                    "message": f"Batch size {current_batch} may be smaller than optimal",
                    "action": f"Consider increasing batch size to {optimal_batch} for better throughput",
                })
        
        # Log results
        logger.info(f"System optimization check complete with {len(results['recommendations'])} recommendations")
        for rec in results["recommendations"]:
            logger.info(f"[{rec['priority']}] {rec['message']} - {rec['action']}")
        
        return results
    
    def generate_report(self, output_path: str = "logs/reliability_report.md"):
        """
        Generate a comprehensive reliability report.
        
        Args:
            output_path: Path to save the report
        """
        logger.info(f"Generating reliability report at {output_path}")
        
        # Collect system info
        system_info = {
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024 ** 3),
            "disk_free_gb": psutil.disk_usage("/").free / (1024 ** 3),
        }
        
        # Run optimization check
        optimization_results = self.system_optimization_check()
        
        # Prepare report
        report = [
            "# HRM Training Reliability Report",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Information",
            f"- OS: {system_info['os']} {system_info['release']} ({system_info['machine']})",
            f"- Python: {system_info['python_version']}",
            f"- PyTorch: {torch.__version__}",
            f"- CPU Cores: {system_info['cpu_count']}",
            f"- Memory: {system_info['memory_gb']:.1f} GB",
            f"- Disk Free: {system_info['disk_free_gb']:.1f} GB",
            f"- MPS Available: {torch.backends.mps.is_available()}",
            "",
            "## Optimization Recommendations",
        ]
        
        # Add recommendations
        for rec in optimization_results["recommendations"]:
            report.append(f"- **[{rec['priority']}]** {rec['message']}")
            report.append(f"  - Action: {rec['action']}")
        
        if not optimization_results["recommendations"]:
            report.append("- No recommendations - system appears optimally configured")
        
        # Add alert history if available
        if self.alert_history:
            report.append("")
            report.append("## Recent Alerts")
            for alert in self.alert_history[-10:]:
                timestamp = datetime.datetime.fromtimestamp(alert["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                report.append(f"- **[{alert['level']}]** {timestamp} - {alert['message']}")
                if "recommendation" in alert:
                    report.append(f"  - Recommendation: {alert['recommendation']}")
        
        # Add training metrics if available
        if self.training_metrics_history:
            report.append("")
            report.append("## Training Metrics Summary")
            
            # Calculate statistics
            steps = [m.step for m in self.training_metrics_history if m.step > 0]
            losses = [m.loss for m in self.training_metrics_history if m.loss is not None]
            throughputs = [m.throughput for m in self.training_metrics_history if m.throughput is not None]
            
            if steps:
                report.append(f"- Steps: {min(steps)} to {max(steps)}")
            if losses:
                report.append(f"- Loss: {np.mean(losses):.4f} (mean), {np.min(losses):.4f} (min), {np.max(losses):.4f} (max)")
            if throughputs:
                report.append(f"- Throughput: {np.mean(throughputs):.2f} samples/sec (mean)")
        
        # Add system metrics if available
        if self.system_metrics_history:
            report.append("")
            report.append("## System Metrics Summary")
            
            # Calculate statistics
            cpu_percents = [m.cpu_percent for m in self.system_metrics_history]
            memory_percents = [m.memory_percent for m in self.system_metrics_history]
            
            report.append(f"- CPU Usage: {np.mean(cpu_percents):.1%} (mean), {np.max(cpu_percents):.1%} (max)")
            report.append(f"- Memory Usage: {np.mean(memory_percents):.1%} (mean), {np.max(memory_percents):.1%} (max)")
            
            if any(m.mps_memory_used is not None for m in self.system_metrics_history):
                mps_memory = [m.mps_memory_used for m in self.system_metrics_history if m.mps_memory_used is not None]
                report.append(f"- MPS Memory: {np.mean(mps_memory):.2f} GB (mean), {np.max(mps_memory):.2f} GB (max)")
        
        # Write report
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write("\n".join(report))
            logger.info(f"Reliability report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save reliability report: {e}")
        
        return report


def main():
    """Main entry point for the reliability monitor."""
    parser = argparse.ArgumentParser(description="HRM Training Reliability Monitor")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="monitor", choices=["monitor", "benchmark", "report"],
                        help="Monitoring mode")
    parser.add_argument("--attach", type=int, help="Attach to an existing process ID")
    parser.add_argument("--alert-threshold", type=float, default=0.9, help="Alert threshold for resource usage")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for benchmarking")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length for benchmarking")
    parser.add_argument("--report-output", type=str, default="logs/reliability_report.md", help="Path for report output")
    
    args = parser.parse_args()
    
    # Update global thresholds
    global CPU_THRESHOLD, DISK_THRESHOLD, MPS_MEMORY_THRESHOLD
    CPU_THRESHOLD = args.alert_threshold
    DISK_THRESHOLD = args.alert_threshold
    MPS_MEMORY_THRESHOLD = args.alert_threshold
    
    # Create monitor
    monitor = ReliabilityMonitor(config_path=args.config, mode=args.mode)
    
    # Handle different modes
    if args.mode == "monitor":
        if args.attach:
            if monitor.attach(args.attach):
                monitor.start()
                try:
                    # Keep running until the process exits or we're interrupted
                    while monitor.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                finally:
                    monitor.stop()
            else:
                sys.exit(1)
        else:
            monitor.start()
            try:
                # Keep running until interrupted
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            finally:
                monitor.stop()
    
    elif args.mode == "benchmark":
        # Import HRM model for benchmarking
        try:
            from hrm.model import HRMModel
            from hrm.config import get_default_mbpp_config
            
            config = get_default_mbpp_config()
            model = HRMModel(config)
            
            monitor.benchmark(
                model=model,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_batches=10,
                warmup=2
            )
        except ImportError as e:
            logger.error(f"Failed to import HRM model: {e}")
            logger.error("Please run this script from the HRM project directory")
            sys.exit(1)
    
    elif args.mode == "report":
        monitor.generate_report(output_path=args.report_output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
