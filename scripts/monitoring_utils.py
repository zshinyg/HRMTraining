#!/usr/bin/env python
"""
Monitoring utilities for HRM training.

This module provides functions for:
- Collecting system metrics (memory, CPU, MPS utilization)
- Detecting training anomalies (loss spikes, gradient explosions)
- Sending alerts via W&B and logging
- Initializing monitoring systems

Designed to be lightweight and work with or without W&B installed.
"""

import logging
import os
import time
from typing import Dict, Optional, Union, Any

import psutil
import torch

# Set up logging
logger = logging.getLogger(__name__)

# Check if wandb is available
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("W&B not installed. Monitoring will use local logging only.")

# Default thresholds for anomaly detection
DEFAULT_THRESHOLDS = {
    "loss": 50.0,  # Loss explosion
    "gradient_norm": 10.0,  # Gradient explosion
    "memory_usage_mb": 14000,  # Approaching M1 limit
    "nan_detected": True,  # NaN values detected
}


def is_wandb_authenticated() -> bool:
    """
    Check if W&B is authenticated either via API key or prior login.

    Returns:
        bool: True if authenticated, False otherwise
    """
    if not WANDB_AVAILABLE:
        return False

    # Check for API key in environment
    if os.environ.get("WANDB_API_KEY"):
        return True

    # Check if already logged in
    try:
        return wandb.api.api_key is not None
    except:
        return False


def collect_system_metrics() -> Dict[str, Union[float, int]]:
    """
    Collect current system metrics including memory usage, CPU and MPS utilization.

    Returns:
        Dict[str, Union[float, int]]: Dictionary of system metrics
    """
    metrics = {}

    # Get current process
    process = psutil.Process(os.getpid())

    # Memory usage in MB
    metrics["memory_usage_mb"] = process.memory_info().rss // (1024 * 1024)

    # CPU utilization percentage
    metrics["cpu_utilization_pct"] = psutil.cpu_percent(interval=None)

    # MPS utilization (Apple Silicon)
    if torch.backends.mps.is_available():
        try:
            # Try to get MPS memory metrics if available
            allocated = (
                torch.mps.current_allocated_memory()
                if hasattr(torch.mps, "current_allocated_memory")
                else 0
            )
            total = (
                torch.mps.driver_allocated_memory()
                if hasattr(torch.mps, "driver_allocated_memory")
                else 1
            )

            # Calculate utilization percentage (avoid division by zero)
            if total > 0:
                metrics["mps_utilization_pct"] = (allocated / total) * 100
            else:
                metrics["mps_utilization_pct"] = 0

            # Also track raw MPS memory in MB
            metrics["mps_allocated_mb"] = allocated // (1024 * 1024)
            metrics["mps_total_mb"] = total // (1024 * 1024)
        except (AttributeError, RuntimeError):
            # Fallback if metrics aren't available
            metrics["mps_utilization_pct"] = 0
            metrics["mps_allocated_mb"] = 0
            metrics["mps_total_mb"] = 0
    else:
        metrics["mps_utilization_pct"] = 0
        metrics["mps_allocated_mb"] = 0
        metrics["mps_total_mb"] = 0

    # Add timestamp
    metrics["timestamp"] = int(time.time())

    # Add samples per second placeholder (to be filled by training loop)
    metrics["samples_per_second"] = 0

    return metrics


def check_anomalies(
    metrics: Dict[str, Any],
    thresholds: Dict[str, float] = DEFAULT_THRESHOLDS,
    step: Optional[int] = None,
) -> bool:
    """
    Check metrics for anomalies based on thresholds and send alerts if needed.

    Args:
        metrics: Dictionary of metrics to check
        thresholds: Dictionary of threshold values
        step: Current training step (for alert context)

    Returns:
        bool: True if anomalies were detected, False otherwise
    """
    anomalies_detected = False
    anomaly_messages = []

    # Check for NaN values in any metric
    has_nan = any(
        isinstance(v, (int, float)) and torch.isnan(torch.tensor(v))
        for v in metrics.values()
    )
    if has_nan:
        msg = "NaN values detected in metrics"
        anomalies_detected = True
        anomaly_messages.append(msg)
        logger.warning(f"ANOMALY ALERT: {msg}")

    # Check each metric against thresholds
    for metric_name, threshold_value in thresholds.items():
        if metric_name in metrics:
            metric_value = metrics[metric_name]

            # Skip non-numeric values
            if not isinstance(metric_value, (int, float)):
                continue

            # For boolean thresholds (like nan_detected)
            if isinstance(threshold_value, bool):
                if metric_value == threshold_value:
                    msg = f"{metric_name} is {metric_value}"
                    anomalies_detected = True
                    anomaly_messages.append(msg)
                    logger.warning(f"ANOMALY ALERT: {msg}")
            # For numeric thresholds
            elif metric_value > threshold_value:
                msg = f"{metric_name} ({metric_value:.2f}) exceeds threshold ({threshold_value:.2f})"
                anomalies_detected = True
                anomaly_messages.append(msg)
                logger.warning(f"ANOMALY ALERT: {msg}")

    # Send alert to W&B if available and authenticated
    if (
        anomalies_detected
        and WANDB_AVAILABLE
        and is_wandb_authenticated()
        and wandb.run is not None
    ):
        step_info = f" at step {step}" if step is not None else ""
        alert_title = f"Training Anomaly Detected{step_info}"
        alert_text = "\n".join(anomaly_messages)

        try:
            wandb.alert(
                title=alert_title, text=alert_text, level=wandb.AlertLevel.ERROR
            )
            logger.info(f"Sent W&B alert: {alert_title}")
        except Exception as e:
            logger.error(f"Failed to send W&B alert: {e}")

    return anomalies_detected


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Log metrics to W&B and local logging.

    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
    """
    # Log to W&B if available
    if WANDB_AVAILABLE and is_wandb_authenticated() and wandb.run is not None:
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics to W&B: {e}")

    # Log to local logger (only key metrics to avoid spam)
    key_metrics = {
        k: v
        for k, v in metrics.items()
        if k in ["loss", "memory_usage_mb", "mps_utilization_pct", "samples_per_second"]
    }
    if key_metrics:
        metrics_str = ", ".join(f"{k}: {v:.2f}" for k, v in key_metrics.items())
        logger.info(f"Metrics: {metrics_str}")


def init_monitoring(
    project_name: str = "hrm-codegen", run_name: str = "m1-27m-training"
) -> bool:
    """
    Initialize monitoring systems including W&B if available.

    Args:
        project_name: W&B project name
        run_name: W&B run name

    Returns:
        bool: True if W&B was successfully initialized, False otherwise
    """
    logger.info("Initializing monitoring systems...")

    wandb_initialized = False

    # Initialize W&B if available and authenticated
    if WANDB_AVAILABLE and is_wandb_authenticated():
        try:
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "monitor_version": "1.0",
                    "hardware": "m1",
                    "thresholds": DEFAULT_THRESHOLDS,
                },
            )
            wandb_initialized = True
            logger.info(
                f"W&B monitoring initialized: project={project_name}, run={run_name}"
            )

            # Set up W&B alert thresholds as custom charts
            wandb.define_metric("loss")
            wandb.define_metric("gradient_norm")
            wandb.define_metric("memory_usage_mb")

            # Define summary metrics
            wandb.define_metric("loss", summary="min")
            wandb.define_metric("memory_usage_mb", summary="max")
            wandb.define_metric("samples_per_second", summary="mean")

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
    else:
        logger.warning(
            "W&B not available or not authenticated. Using local logging only."
        )

    # Log initial system metrics
    initial_metrics = collect_system_metrics()
    logger.info(
        f"Initial system metrics: Memory={initial_metrics['memory_usage_mb']}MB, "
        f"CPU={initial_metrics['cpu_utilization_pct']}%, "
        f"MPS={initial_metrics['mps_utilization_pct']:.1f}%"
    )

    return wandb_initialized


def create_custom_charts() -> None:
    """
    Create custom charts in W&B for better visualization.
    Only called if W&B is available and initialized.
    """
    if not (WANDB_AVAILABLE and is_wandb_authenticated() and wandb.run is not None):
        return

    try:
        # Create memory usage panel
        wandb.log(
            {
                "memory_usage": wandb.plot.line_series(
                    xs=[[0]],
                    ys=[[0]],
                    keys=["Memory Usage (MB)"],
                    title="Memory Usage",
                    xname="Step",
                )
            }
        )

        # Create MPS utilization gauge
        wandb.log(
            {
                "mps_gauge": wandb.plot.gauge(
                    value=0, title="MPS Utilization", min_value=0, max_value=100
                )
            }
        )

        # Create training throughput chart
        wandb.log(
            {
                "throughput": wandb.plot.line_series(
                    xs=[[0]],
                    ys=[[0]],
                    keys=["Samples/Second"],
                    title="Training Throughput",
                    xname="Step",
                )
            }
        )

        logger.info("Created custom W&B charts")
    except Exception as e:
        logger.error(f"Failed to create custom W&B charts: {e}")


def finish_monitoring() -> None:
    """Clean up monitoring resources and finalize logs."""
    if WANDB_AVAILABLE and wandb.run is not None:
        try:
            wandb.finish()
            logger.info("W&B monitoring finalized")
        except Exception as e:
            logger.error(f"Error finalizing W&B: {e}")
