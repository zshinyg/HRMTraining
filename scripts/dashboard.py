#!/usr/bin/env python
"""
Comprehensive Training Dashboard

This script provides a web-based dashboard for monitoring, visualizing, and managing
the entire training infrastructure, including real-time status monitoring, training
progress tracking, system resource usage, health checks, failure analysis, CI/CD
pipeline status, and experiment comparison.

Features:
- Real-time status monitoring for all infrastructure components
- Training progress visualization with key metrics and learning curves
- System resource usage monitoring (CPU, GPU, memory, disk, network)
- Health check status and failure analysis with recovery options
- CI/CD pipeline status and job information
- Cost tracking and resource optimization recommendations
- Experiment comparison and historical analysis
- Interactive controls for training management and recovery
- Automated alerts and notifications
- Integration with external monitoring systems (W&B, TensorBoard, Prometheus, Grafana)
- Export of status data and report generation in multiple formats

Usage:
    # Start the dashboard
    python scripts/dashboard.py --port 8501 --config configs/training_config.yaml
    
    # Start with specific experiment
    python scripts/dashboard.py --experiment-name hrm_training --output-dir outputs
    
    # Start in read-only mode (no control capabilities)
    python scripts/dashboard.py --read-only
    
    # Start with external monitoring integration
    python scripts/dashboard.py --enable-wandb --enable-prometheus
"""

import argparse
import datetime
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
import tempfile
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dashboard")

# Optional imports with fallbacks
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit not available. Install with: pip install streamlit")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install with: pip install plotly")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available. Install with: pip install pandas")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")

try:
    import wandb
    from wandb.integration.streamlit import wandb_chart
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not available. Install with: pip install wandb")

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Install with: pip install psutil")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available. Install with: pip install requests")

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False
    logger.warning("Altair not available. Install with: pip install altair")

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False
    logger.warning("PyDeck not available. Install with: pip install pydeck")

try:
    from prometheus_client.parser import text_string_to_metric_families
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Install with: pip install prometheus-client")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Install with: pip install matplotlib")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Install with: pip install pillow")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available. Install with: pip install seaborn")

# Default configuration values
DEFAULT_CONFIG = {
    "dashboard": {
        "title": "HRM Training Dashboard",
        "refresh_interval_seconds": 10,
        "port": 8501,
        "theme": "light",  # light or dark
        "show_sidebar": True,
        "auto_refresh": True,
        "read_only": False,
        "enable_wandb": True,
        "enable_tensorboard": True,
        "enable_prometheus": False,
        "enable_grafana": False,
        "prometheus_url": "http://localhost:9090",
        "grafana_url": "http://localhost:3000",
        "notification_channels": ["slack", "email"],
        "slack_webhook_url": "",
        "email_recipients": [],
        "alert_levels": ["critical", "warning"],
        "export_formats": ["csv", "json", "html", "pdf"],
        "max_experiments": 10,
        "default_metrics": ["loss", "accuracy", "learning_rate"],
        "system_metrics": ["cpu", "memory", "gpu", "disk", "network"],
        "default_view": "overview",
    },
    "experiment_name": "hrm_training",
    "output_dir": "outputs",
    "data_dir": "data",
}


class DashboardConfig:
    """Configuration for the training dashboard."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        port: Optional[int] = None,
        read_only: Optional[bool] = None,
        enable_wandb: Optional[bool] = None,
        enable_prometheus: Optional[bool] = None,
    ):
        """
        Initialize dashboard configuration.
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name of the experiment (overrides config)
            output_dir: Directory containing experiment outputs (overrides config)
            port: Port for the dashboard server (overrides config)
            read_only: Whether the dashboard is read-only (overrides config)
            enable_wandb: Whether to enable W&B integration (overrides config)
            enable_prometheus: Whether to enable Prometheus integration (overrides config)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Override config with constructor arguments
        if experiment_name:
            self.config["experiment_name"] = experiment_name
        
        if output_dir:
            self.config["output_dir"] = output_dir
        
        if port:
            self.config["dashboard"]["port"] = port
        
        if read_only is not None:
            self.config["dashboard"]["read_only"] = read_only
        
        if enable_wandb is not None:
            self.config["dashboard"]["enable_wandb"] = enable_wandb
        
        if enable_prometheus is not None:
            self.config["dashboard"]["enable_prometheus"] = enable_prometheus
        
        # Extract commonly used values
        self.experiment_name = self.config["experiment_name"]
        self.output_dir = Path(self.config["output_dir"])
        self.data_dir = Path(self.config.get("data_dir", "data"))
        self.dashboard_config = self.config["dashboard"]
        self.title = self.dashboard_config["title"]
        self.refresh_interval = self.dashboard_config["refresh_interval_seconds"]
        self.port = self.dashboard_config["port"]
        self.theme = self.dashboard_config["theme"]
        self.read_only = self.dashboard_config["read_only"]
        self.auto_refresh = self.dashboard_config["auto_refresh"]
        self.enable_wandb = self.dashboard_config["enable_wandb"] and WANDB_AVAILABLE
        self.enable_tensorboard = self.dashboard_config["enable_tensorboard"] and TENSORBOARD_AVAILABLE
        self.enable_prometheus = self.dashboard_config["enable_prometheus"] and PROMETHEUS_AVAILABLE
        self.enable_grafana = self.dashboard_config["enable_grafana"]
        self.prometheus_url = self.dashboard_config["prometheus_url"]
        self.grafana_url = self.dashboard_config["grafana_url"]
        self.notification_channels = self.dashboard_config["notification_channels"]
        self.slack_webhook_url = self.dashboard_config["slack_webhook_url"]
        self.email_recipients = self.dashboard_config["email_recipients"]
        self.alert_levels = self.dashboard_config["alert_levels"]
        self.export_formats = self.dashboard_config["export_formats"]
        self.max_experiments = self.dashboard_config["max_experiments"]
        self.default_metrics = self.dashboard_config["default_metrics"]
        self.system_metrics = self.dashboard_config["system_metrics"]
        self.default_view = self.dashboard_config["default_view"]
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self.config
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration (default: output_dir/dashboard_config.yaml)
            
        Returns:
            str: Path to saved configuration file
        """
        if path is None:
            path = str(self.output_dir / "dashboard_config.yaml")
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Saved dashboard configuration to {path}")
            return path
        
        except Exception as e:
            logger.error(f"Error saving dashboard configuration: {e}")
            return ""


class ExperimentData:
    """Data manager for experiment data and metrics."""
    
    def __init__(self, config: DashboardConfig):
        """
        Initialize experiment data manager.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        self.experiment_name = config.experiment_name
        self.output_dir = config.output_dir
        self.data_dir = config.data_dir
        
        # Initialize data storage
        self.training_metrics = {}
        self.system_metrics = {}
        self.health_checks = {}
        self.failures = {}
        self.checkpoints = {}
        self.ci_jobs = {}
        self.experiments = {}
        
        # Cache for expensive operations
        self.cache = {}
        self.cache_timestamp = {}
        self.cache_ttl = 60  # seconds
        
        # Last update timestamp
        self.last_update = 0
        
        # Initialize W&B integration
        self.wandb_run = None
        if config.enable_wandb:
            self._init_wandb()
        
        # Load initial data
        self.update()
    
    def _init_wandb(self):
        """Initialize W&B integration."""
        if not WANDB_AVAILABLE:
            return
        
        try:
            # Check if API key is set
            if "WANDB_API_KEY" not in os.environ:
                logger.warning("W&B API key not set. Set the WANDB_API_KEY environment variable.")
                return
            
            # Initialize W&B in read-only mode
            wandb.init(
                project=self.experiment_name,
                name=f"dashboard-{self.experiment_name}",
                job_type="dashboard",
                dir=str(self.output_dir / "wandb"),
                mode="online" if not self.config.read_only else "offline",
            )
            
            self.wandb_run = wandb.run
            logger.info(f"Initialized W&B integration for project: {self.experiment_name}")
        
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            self.wandb_run = None
    
    def update(self) -> bool:
        """
        Update all experiment data.
        
        Returns:
            bool: Whether the update was successful
        """
        try:
            # Update timestamp
            self.last_update = time.time()
            
            # Update training metrics
            self._update_training_metrics()
            
            # Update system metrics
            self._update_system_metrics()
            
            # Update health checks
            self._update_health_checks()
            
            # Update failures
            self._update_failures()
            
            # Update checkpoints
            self._update_checkpoints()
            
            # Update CI jobs
            self._update_ci_jobs()
            
            # Update experiments
            self._update_experiments()
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating experiment data: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def _update_training_metrics(self):
        """Update training metrics data."""
        # Check for metrics files
        metrics_file = self.output_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    self.training_metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics from {metrics_file}: {e}")
        
        # Check for TensorBoard logs
        if TENSORBOARD_AVAILABLE and self.config.enable_tensorboard:
            tensorboard_dir = self.output_dir / "tensorboard"
            if tensorboard_dir.exists():
                try:
                    # Find the latest event file
                    event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
                    if event_files:
                        latest_file = max(event_files, key=lambda f: f.stat().st_mtime)
                        
                        # Load events
                        ea = event_accumulator.EventAccumulator(str(latest_file))
                        ea.Reload()
                        
                        # Get scalar metrics
                        for tag in ea.Tags()["scalars"]:
                            events = ea.Scalars(tag)
                            if events:
                                # Convert to list of (step, value) pairs
                                values = [(e.step, e.value) for e in events]
                                
                                # Add to metrics
                                if "tensorboard" not in self.training_metrics:
                                    self.training_metrics["tensorboard"] = {}
                                
                                self.training_metrics["tensorboard"][tag] = values
                
                except Exception as e:
                    logger.error(f"Error loading TensorBoard events: {e}")
        
        # Get metrics from W&B
        if WANDB_AVAILABLE and self.config.enable_wandb and self.wandb_run:
            try:
                # Get history
                api = wandb.Api()
                runs = api.runs(f"{wandb.run.entity}/{wandb.run.project}")
                
                if runs:
                    # Get the latest run
                    latest_run = runs[0]
                    history = latest_run.history()
                    
                    # Convert to dictionary
                    if not history.empty:
                        wandb_metrics = {}
                        for column in history.columns:
                            if column not in ["_step", "_runtime", "_timestamp"]:
                                values = list(zip(history["_step"], history[column]))
                                wandb_metrics[column] = values
                        
                        # Add to metrics
                        self.training_metrics["wandb"] = wandb_metrics
            
            except Exception as e:
                logger.error(f"Error loading W&B metrics: {e}")
    
    def _update_system_metrics(self):
        """Update system metrics data."""
        # Check for system metrics files
        metrics_file = self.output_dir / "system_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    self.system_metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error loading system metrics from {metrics_file}: {e}")
        
        # Get current system metrics
        if PSUTIL_AVAILABLE:
            try:
                current_metrics = {
                    "timestamp": time.time(),
                    "cpu": {
                        "percent": psutil.cpu_percent(interval=0.1),
                        "count": psutil.cpu_count(),
                    },
                    "memory": {
                        "total": psutil.virtual_memory().total / (1024 ** 3),  # GB
                        "available": psutil.virtual_memory().available / (1024 ** 3),  # GB
                        "used": psutil.virtual_memory().used / (1024 ** 3),  # GB
                        "percent": psutil.virtual_memory().percent,
                    },
                    "disk": {
                        "total": psutil.disk_usage("/").total / (1024 ** 3),  # GB
                        "free": psutil.disk_usage("/").free / (1024 ** 3),  # GB
                        "used": psutil.disk_usage("/").used / (1024 ** 3),  # GB
                        "percent": psutil.disk_usage("/").percent,
                    },
                }
                
                # Add GPU metrics if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    current_metrics["gpu"] = []
                    for i in range(torch.cuda.device_count()):
                        gpu_metrics = {
                            "id": i,
                            "name": torch.cuda.get_device_name(i),
                            "memory_allocated": torch.cuda.memory_allocated(i) / (1024 ** 3),  # GB
                            "memory_reserved": torch.cuda.memory_reserved(i) / (1024 ** 3),  # GB
                        }
                        current_metrics["gpu"].append(gpu_metrics)
                
                # Add MPS metrics if available
                if TORCH_AVAILABLE and hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    current_metrics["mps"] = {
                        "available": True,
                    }
                
                # Add to metrics history
                if "history" not in self.system_metrics:
                    self.system_metrics["history"] = []
                
                self.system_metrics["history"].append(current_metrics)
                
                # Limit history size
                max_history = 1000
                if len(self.system_metrics["history"]) > max_history:
                    self.system_metrics["history"] = self.system_metrics["history"][-max_history:]
                
                # Set current metrics
                self.system_metrics["current"] = current_metrics
            
            except Exception as e:
                logger.error(f"Error getting current system metrics: {e}")
        
        # Get metrics from Prometheus if enabled
        if PROMETHEUS_AVAILABLE and self.config.enable_prometheus:
            try:
                # Query Prometheus
                response = requests.get(f"{self.config.prometheus_url}/api/v1/query", params={
                    "query": "up"  # Simple query to check if Prometheus is up
                }, timeout=5)
                
                if response.status_code == 200:
                    # Prometheus is up, get metrics
                    metrics_response = requests.get(f"{self.config.prometheus_url}/metrics", timeout=5)
                    if metrics_response.status_code == 200:
                        # Parse metrics
                        prometheus_metrics = {}
                        for family in text_string_to_metric_families(metrics_response.text):
                            for sample in family.samples:
                                # Add to metrics
                                metric_name = sample.name
                                metric_value = sample.value
                                metric_labels = sample.labels
                                
                                if metric_name not in prometheus_metrics:
                                    prometheus_metrics[metric_name] = []
                                
                                prometheus_metrics[metric_name].append({
                                    "value": metric_value,
                                    "labels": metric_labels,
                                    "timestamp": time.time(),
                                })
                        
                        # Add to system metrics
                        self.system_metrics["prometheus"] = prometheus_metrics
            
            except Exception as e:
                logger.error(f"Error getting Prometheus metrics: {e}")
    
    def _update_health_checks(self):
        """Update health check data."""
        # Check for health check files
        health_dir = self.output_dir / "health"
        if health_dir.exists():
            try:
                # Check for health check results
                health_file = health_dir / "health_check.json"
                if health_file.exists():
                    with open(health_file, "r") as f:
                        self.health_checks["latest"] = json.load(f)
                
                # Check for heartbeat file
                heartbeat_file = health_dir / "HEARTBEAT"
                if heartbeat_file.exists():
                    with open(heartbeat_file, "r") as f:
                        lines = f.readlines()
                        heartbeat = {}
                        for line in lines:
                            if ":" in line:
                                key, value = line.strip().split(":", 1)
                                heartbeat[key.strip()] = value.strip()
                        
                        self.health_checks["heartbeat"] = heartbeat
                
                # Check for health check history
                history_files = list(health_dir.glob("health_check_*.json"))
                if history_files:
                    history = []
                    for file in sorted(history_files, key=lambda f: f.stat().st_mtime):
                        with open(file, "r") as f:
                            history.append(json.load(f))
                    
                    self.health_checks["history"] = history
            
            except Exception as e:
                logger.error(f"Error loading health checks: {e}")
    
    def _update_failures(self):
        """Update failure data."""
        # Check for failure analysis directory
        failure_dir = self.output_dir / "failure_analysis"
        if failure_dir.exists():
            try:
                # Check for failure data
                data_file = failure_dir / "data" / "failures.pkl"
                if data_file.exists():
                    # We can't directly load pickle files for security reasons
                    # Instead, check for JSON exports
                    json_file = failure_dir / "data" / "failures.json"
                    if json_file.exists():
                        with open(json_file, "r") as f:
                            self.failures["data"] = json.load(f)
                
                # Check for failure reports
                report_dir = failure_dir / "reports"
                if report_dir.exists():
                    reports = []
                    for file in sorted(report_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
                        with open(file, "r") as f:
                            reports.append(json.load(f))
                    
                    if reports:
                        self.failures["reports"] = reports
                        self.failures["latest_report"] = reports[0]
            
            except Exception as e:
                logger.error(f"Error loading failure data: {e}")
    
    def _update_checkpoints(self):
        """Update checkpoint data."""
        # Check for checkpoints directory
        checkpoint_dir = self.output_dir / "checkpoints"
        if checkpoint_dir.exists():
            try:
                # Get checkpoint files
                checkpoint_files = list(checkpoint_dir.glob("*.pt"))
                
                if checkpoint_files:
                    checkpoints = []
                    for file in sorted(checkpoint_files, key=lambda f: f.stat().st_mtime, reverse=True):
                        checkpoint = {
                            "filename": file.name,
                            "path": str(file),
                            "size_mb": file.stat().st_size / (1024 * 1024),
                            "modified_time": file.stat().st_mtime,
                            "modified_date": datetime.datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        
                        # Check for metadata file
                        metadata_file = checkpoint_dir / f"{file.stem}_metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, "r") as f:
                                checkpoint["metadata"] = json.load(f)
                        
                        checkpoints.append(checkpoint)
                    
                    self.checkpoints["files"] = checkpoints
                    
                    # Get latest checkpoint
                    if checkpoints:
                        self.checkpoints["latest"] = checkpoints[0]
                        
                        # Check for best checkpoint
                        best_checkpoints = [c for c in checkpoints if "best" in c["filename"]]
                        if best_checkpoints:
                            self.checkpoints["best"] = best_checkpoints[0]
            
            except Exception as e:
                logger.error(f"Error loading checkpoint data: {e}")
    
    def _update_ci_jobs(self):
        """Update CI/CD job data."""
        # Check for CI/CD job files
        ci_dir = self.output_dir / "ci"
        if ci_dir.exists():
            try:
                # Get job files
                job_files = list(ci_dir.glob("job_*.json"))
                
                if job_files:
                    jobs = []
                    for file in sorted(job_files, key=lambda f: f.stat().st_mtime, reverse=True):
                        with open(file, "r") as f:
                            jobs.append(json.load(f))
                    
                    self.ci_jobs["jobs"] = jobs
                    
                    # Get latest job
                    if jobs:
                        self.ci_jobs["latest"] = jobs[0]
                
                # Check for CI/CD status file
                status_file = ci_dir / "status.json"
                if status_file.exists():
                    with open(status_file, "r") as f:
                        self.ci_jobs["status"] = json.load(f)
            
            except Exception as e:
                logger.error(f"Error loading CI/CD job data: {e}")
        
        # Check for GitHub Actions workflow runs
        if "GITHUB_REPOSITORY" in os.environ and REQUESTS_AVAILABLE:
            try:
                # Get GitHub token
                github_token = os.environ.get("GITHUB_TOKEN")
                if github_token:
                    # Get repository
                    repo = os.environ["GITHUB_REPOSITORY"]
                    
                    # Get workflow runs
                    headers = {
                        "Authorization": f"token {github_token}",
                        "Accept": "application/vnd.github.v3+json",
                    }
                    
                    response = requests.get(
                        f"https://api.github.com/repos/{repo}/actions/runs",
                        headers=headers,
                    timeout=10)
                    
                    if response.status_code == 200:
                        workflow_runs = response.json()
                        
                        # Add to CI jobs
                        self.ci_jobs["github_actions"] = workflow_runs
            
            except Exception as e:
                logger.error(f"Error loading GitHub Actions workflow runs: {e}")
    
    def _update_experiments(self):
        """Update experiment data."""
        # Check for experiment directories
        experiment_dirs = []
        for item in self.output_dir.parent.iterdir():
            if item.is_dir() and item.name != self.experiment_name:
                # Check if this is an experiment directory
                if (item / "config.yaml").exists() or (item / "metrics.json").exists():
                    experiment_dirs.append(item)
        
        if experiment_dirs:
            experiments = {}
            
            for exp_dir in sorted(experiment_dirs, key=lambda d: d.stat().st_mtime, reverse=True):
                exp_name = exp_dir.name
                
                # Skip if we already have too many experiments
                if len(experiments) >= self.config.max_experiments:
                    break
                
                # Get experiment data
                experiment = {
                    "name": exp_name,
                    "path": str(exp_dir),
                    "modified_time": exp_dir.stat().st_mtime,
                    "modified_date": datetime.datetime.fromtimestamp(exp_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
                
                # Check for config file
                config_file = exp_dir / "config.yaml"
                if config_file.exists():
                    try:
                        with open(config_file, "r") as f:
                            experiment["config"] = yaml.safe_load(f)
                    except Exception:
                        pass
                
                # Check for metrics file
                metrics_file = exp_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file, "r") as f:
                            experiment["metrics"] = json.load(f)
                    except Exception:
                        pass
                
                # Add to experiments
                experiments[exp_name] = experiment
            
            self.experiments = experiments
    
    def get_training_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get training metrics.
        
        Args:
            metric_names: List of metric names to retrieve (default: all)
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        if not metric_names:
            return self.training_metrics
        
        # Filter metrics by name
        filtered_metrics = {}
        
        for source, metrics in self.training_metrics.items():
            filtered_source = {}
            
            for name, values in metrics.items():
                if any(metric in name for metric in metric_names):
                    filtered_source[name] = values
            
            if filtered_source:
                filtered_metrics[source] = filtered_source
        
        return filtered_metrics
    
    def get_system_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get system metrics.
        
        Args:
            metric_names: List of metric names to retrieve (default: all)
            
        Returns:
            Dict[str, Any]: System metrics
        """
        if not metric_names:
            return self.system_metrics
        
        # Filter metrics by name
        if "history" in self.system_metrics:
            filtered_history = []
            
            for entry in self.system_metrics["history"]:
                filtered_entry = {"timestamp": entry["timestamp"]}
                
                for name in metric_names:
                    if name in entry:
                        filtered_entry[name] = entry[name]
                
                filtered_history.append(filtered_entry)
            
            return {"history": filtered_history, "current": self.system_metrics.get("current", {})}
        
        return self.system_metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status.
        
        Returns:
            Dict[str, Any]: Health status
        """
        return self.health_checks
    
    def get_failures(self) -> Dict[str, Any]:
        """
        Get failures.
        
        Returns:
            Dict[str, Any]: Failures
        """
        return self.failures
    
    def get_checkpoints(self) -> Dict[str, Any]:
        """
        Get checkpoints.
        
        Returns:
            Dict[str, Any]: Checkpoints
        """
        return self.checkpoints
    
    def get_ci_jobs(self) -> Dict[str, Any]:
        """
        Get CI/CD jobs.
        
        Returns:
            Dict[str, Any]: CI/CD jobs
        """
        return self.ci_jobs
    
    def get_experiments(self) -> Dict[str, Any]:
        """
        Get experiments.
        
        Returns:
            Dict[str, Any]: Experiments
        """
        return self.experiments
    
    def get_experiment_comparison(self, experiment_names: List[str], metric_names: List[str]) -> Dict[str, Any]:
        """
        Get experiment comparison.
        
        Args:
            experiment_names: List of experiment names to compare
            metric_names: List of metric names to compare
            
        Returns:
            Dict[str, Any]: Experiment comparison
        """
        comparison = {
            "experiments": experiment_names,
            "metrics": metric_names,
            "data": {},
        }
        
        # Add current experiment
        experiments = dict(self.experiments)
        experiments[self.experiment_name] = {
            "name": self.experiment_name,
            "path": str(self.output_dir),
            "metrics": self.training_metrics,
        }
        
        # Filter experiments
        filtered_experiments = {}
        for name in experiment_names:
            if name in experiments:
                filtered_experiments[name] = experiments[name]
        
        # Extract metrics
        for exp_name, exp_data in filtered_experiments.items():
            comparison["data"][exp_name] = {}
            
            if "metrics" in exp_data:
                metrics = exp_data["metrics"]
                
                for source, source_metrics in metrics.items():
                    for metric_name, values in source_metrics.items():
                        if any(name in metric_name for name in metric_names):
                            comparison["data"][exp_name][metric_name] = values
        
        return comparison
    
    def get_cost_tracking(self) -> Dict[str, Any]:
        """
        Get cost tracking data.
        
        Returns:
            Dict[str, Any]: Cost tracking data
        """
        # Check for cost tracking file
        cost_file = self.output_dir / "cost_tracking.json"
        if cost_file.exists():
            try:
                with open(cost_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cost tracking data: {e}")
        
        return {}
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get optimization recommendations.
        
        Returns:
            List[Dict[str, Any]]: Optimization recommendations
        """
        # Check for recommendations file
        recommendations_file = self.output_dir / "recommendations.json"
        if recommendations_file.exists():
            try:
                with open(recommendations_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading optimization recommendations: {e}")
        
        return []
    
    def export_data(self, data_type: str, format: str, path: Optional[str] = None) -> str:
        """
        Export data to file.
        
        Args:
            data_type: Type of data to export (training, system, health, failures, checkpoints, ci, experiments)
            format: Export format (csv, json, html, pdf)
            path: Path to save exported data (default: output_dir/exports/{data_type}.{format})
            
        Returns:
            str: Path to exported data file
        """
        if path is None:
            # Create exports directory
            exports_dir = self.output_dir / "exports"
            exports_dir.mkdir(exist_ok=True)
            
            path = str(exports_dir / f"{data_type}.{format}")
        
        try:
            # Get data to export
            data = None
            if data_type == "training":
                data = self.training_metrics
            elif data_type == "system":
                data = self.system_metrics
            elif data_type == "health":
                data = self.health_checks
            elif data_type == "failures":
                data = self.failures
            elif data_type == "checkpoints":
                data = self.checkpoints
            elif data_type == "ci":
                data = self.ci_jobs
            elif data_type == "experiments":
                data = self.experiments
            elif data_type == "all":
                data = {
                    "training": self.training_metrics,
                    "system": self.system_metrics,
                    "health": self.health_checks,
                    "failures": self.failures,
                    "checkpoints": self.checkpoints,
                    "ci": self.ci_jobs,
                    "experiments": self.experiments,
                }
            else:
                logger.error(f"Unknown data type: {data_type}")
                return ""
            
            # Export data
            if format == "json":
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
            
            elif format == "csv" and PANDAS_AVAILABLE:
                # Convert to DataFrame
                if data_type == "training":
                    # Flatten training metrics
                    flat_data = []
                    for source, metrics in data.items():
                        for metric_name, values in metrics.items():
                            for step, value in values:
                                flat_data.append({
                                    "source": source,
                                    "metric": metric_name,
                                    "step": step,
                                    "value": value,
                                })
                    
                    df = pd.DataFrame(flat_data)
                
                elif data_type == "system" and "history" in data:
                    # Flatten system metrics history
                    flat_data = []
                    for entry in data["history"]:
                        flat_entry = {"timestamp": entry["timestamp"]}
                        
                        # Flatten nested dictionaries
                        for key, value in entry.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    flat_entry[f"{key}_{subkey}"] = subvalue
                            else:
                                flat_entry[key] = value
                        
                        flat_data.append(flat_entry)
                    
                    df = pd.DataFrame(flat_data)
                
                elif data_type == "checkpoints" and "files" in data:
                    df = pd.DataFrame(data["files"])
                
                else:
                    # Generic conversion
                    df = pd.DataFrame([data])
                
                # Save to CSV
                df.to_csv(path, index=False)
            
            elif format == "html" and PANDAS_AVAILABLE:
                # Convert to HTML
                if data_type == "training":
                    # Create HTML with plotly
                    if PLOTLY_AVAILABLE:
                        # Create figures for each metric
                        figures = []
                        for source, metrics in data.items():
                            for metric_name, values in metrics.items():
                                if values:
                                    steps, values = zip(*values)
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=steps,
                                        y=values,
                                        mode="lines",
                                        name=metric_name,
                                    ))
                                    fig.update_layout(
                                        title=f"{metric_name} ({source})",
                                        xaxis_title="Step",
                                        yaxis_title="Value",
                                    )
                                    figures.append(fig)
                        
                        # Combine figures into HTML
                        html = "<html><head><title>Training Metrics</title></head><body>"
                        html += f"<h1>Training Metrics - {self.experiment_name}</h1>"
                        
                        for fig in figures:
                            html += fig.to_html(full_html=False)
                        
                        html += "</body></html>"
                        
                        with open(path, "w") as f:
                            f.write(html)
                    else:
                        # Fallback to pandas HTML
                        flat_data = []
                        for source, metrics in data.items():
                            for metric_name, values in metrics.items():
                                for step, value in values:
                                    flat_data.append({
                                        "source": source,
                                        "metric": metric_name,
                                        "step": step,
                                        "value": value,
                                    })
                        
                        df = pd.DataFrame(flat_data)
                        html = df.to_html()
                        
                        with open(path, "w") as f:
                            f.write(html)
                
                elif data_type == "system" and "history" in data:
                    # Flatten system metrics history
                    flat_data = []
                    for entry in data["history"]:
                        flat_entry = {"timestamp": entry["timestamp"]}
                        
                        # Flatten nested dictionaries
                        for key, value in entry.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    flat_entry[f"{key}_{subkey}"] = subvalue
                            else:
                                flat_entry[key] = value
                        
                        flat_data.append(flat_entry)
                    
                    df = pd.DataFrame(flat_data)
                    html = df.to_html()
                    
                    with open(path, "w") as f:
                        f.write(html)
                
                else:
                    # Generic conversion
                    df = pd.DataFrame([data])
                    html = df.to_html()
                    
                    with open(path, "w") as f:
                        f.write(html)
            
            elif format == "pdf":
                # PDF export requires additional libraries
                logger.warning("PDF export not implemented yet")
                return ""
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return ""
            
            logger.info(f"Exported {data_type} data to {path}")
            return path
        
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return ""


class TrainingControl:
    """Control interface for training management."""
    
    def __init__(self, config: DashboardConfig):
        """
        Initialize training control.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        self.experiment_name = config.experiment_name
        self.output_dir = config.output_dir
        self.read_only = config.read_only
        
        # Initialize orchestrator reference
        self.orchestrator = None
        self._load_orchestrator()
    
    def _load_orchestrator(self):
        """Load training orchestrator."""
        if self.read_only:
            logger.info("Dashboard is in read-only mode, training control disabled")
            return
        
        try:
            # Import orchestrator
            sys.path.insert(0, ".")
            from scripts.training.training_orchestrator import TrainingOrchestrator
            
            # Check if orchestrator is already running
            status_file = self.output_dir / "training_status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r") as f:
                        status = json.load(f)
                    
                    if status.get("status") == "running" and status.get("pid"):
                        # Check if process is still running
                        pid = status["pid"]
                        try:
                            os.kill(pid, 0)  # This will raise an exception if process is not running
                            logger.info(f"Training orchestrator is running with PID {pid}")
                            
                            # TODO: Implement RPC to communicate with running orchestrator
                            return
                        except OSError:
                            # Process is not running
                            logger.info(f"Training orchestrator with PID {pid} is not running")
                
                except Exception as e:
                    logger.error(f"Error checking training status: {e}")
            
            # Initialize new orchestrator
            self.orchestrator = TrainingOrchestrator(
                experiment_name=self.experiment_name,
                output_dir=str(self.output_dir),
                read_only=True,  # Initialize in read-only mode for safety
            )
            
            logger.info("Initialized training orchestrator in read-only mode")
        
        except Exception as e:
            logger.error(f"Error loading training orchestrator: {e}")
    
    def start_training(self, config_path: Optional[str] = None) -> bool:
        """
        Start training.
        
        Args:
            config_path: Path to configuration file (default: use existing config)
            
        Returns:
            bool: Whether the operation was successful
        """
        if self.read_only:
            logger.warning("Dashboard is in read-only mode, cannot start training")
            return False
        
        try:
            # Check if training is already running
            status = self.get_training_status()
            if status.get("status") == "running":
                logger.warning("Training is already running")
                return False
            
            # Start training in a separate process
            cmd = [
                sys.executable,
                "-m",
                "scripts.training.training_orchestrator",
                "--experiment-name",
                self.experiment_name,
                "--output-dir",
                str(self.output_dir),
            ]
            
            if config_path:
                cmd.extend(["--config-path", config_path])
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            logger.info(f"Started training process with PID {process.pid}")
            
            # Wait a bit to check if process started successfully
            time.sleep(2)
            
            if process.poll() is not None:
                # Process exited
                stdout, stderr = process.communicate()
                logger.error(f"Training process exited with code {process.returncode}")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return False
    
    def stop_training(self) -> bool:
        """
        Stop training.
        
        Returns:
            bool: Whether the operation was successful
        """
        if self.read_only:
            logger.warning("Dashboard is in read-only mode, cannot stop training")
            return False
        
        try:
            # Check if training is running
            status = self.get_training_status()
            if status.get("status") != "running" or not status.get("pid"):
                logger.warning("Training is not running")
                return False
            
            # Send signal to stop training
            pid = status["pid"]
            os.kill(pid, signal.SIGTERM)
            
            logger.info(f"Sent SIGTERM to training process with PID {pid}")
            
            # Wait for process to exit
            for _ in range(10):
                try:
                    os.kill(pid, 0)  # This will raise an exception if process is not running
                    time.sleep(1)
                except OSError:
                    # Process has exited
                    logger.info(f"Training process with PID {pid} has exited")
                    return True
            
            # Process didn't exit, try SIGKILL
            try:
                os.kill(pid, signal.SIGKILL)
                logger.warning(f"Sent SIGKILL to training process with PID {pid}")
                return True
            except OSError:
                # Process has already exited
                return True
        
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return False
    
    def pause_training(self) -> bool:
        """
        Pause training.
        
        Returns:
            bool: Whether the operation was successful
        """
        if self.read_only:
            logger.warning("Dashboard is in read-only mode, cannot pause training")
            return False
        
        try:
            # Check if training is running
            status = self.get_training_status()
            if status.get("status") != "running" or not status.get("pid"):
                logger.warning("Training is not running")
                return False
            
            # Create pause signal file
            pause_file = self.output_dir / "PAUSE_TRAINING"
            with open(pause_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"dashboard_pid: {os.getpid()}\n")
            
            logger.info(f"Created pause signal file: {pause_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error pausing training: {e}")
            return False
    
    def resume_training(self) -> bool:
        """
        Resume training.
        
        Returns:
            bool: Whether the operation was successful
        """
        if self.read_only:
            logger.warning("Dashboard is in read-only mode, cannot resume training")
            return False
        
        try:
            # Check if training is paused
            status = self.get_training_status()
            if status.get("status") != "paused":
                logger.warning("Training is not paused")
                return False
            
            # Remove pause signal file
            pause_file = self.output_dir / "PAUSE_TRAINING"
            if pause_file.exists():
                pause_file.unlink()
            
            # Create resume signal file
            resume_file = self.output_dir / "RESUME_TRAINING"
            with open(resume_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"dashboard_pid: {os.getpid()}\n")
            
            logger.info(f"Created resume signal file: {resume_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error resuming training: {e}")
            return False
    
    def save_checkpoint(self, tag: Optional[str] = None) -> bool:
        """
        Save checkpoint.
        
        Args:
            tag: Checkpoint tag (default: manual)
            
        Returns:
            bool: Whether the operation was successful
        """
        if self.read_only:
            logger.warning("Dashboard is in read-only mode, cannot save checkpoint")
            return False
        
        try:
            # Check if training is running
            status = self.get_training_status()
            if status.get("status") != "running" and status.get("status") != "paused":
                logger.warning("Training is not running or paused")
                return False
            
            # Create checkpoint signal file
            checkpoint_file = self.output_dir / "SAVE_CHECKPOINT"
            with open(checkpoint_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"dashboard_pid: {os.getpid()}\n")
                if tag:
                    f.write(f"tag: {tag}\n")
                else:
                    f.write("tag: manual\n")
            
            logger.info(f"Created checkpoint signal file: {checkpoint_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            bool: Whether the operation was successful
        """
        if self.read_only:
            logger.warning("Dashboard is in read-only mode, cannot load checkpoint")
            return False
        
        try:
            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            # Check if training is running
            status = self.get_training_status()
            if status.get("status") == "running":
                logger.warning("Cannot load checkpoint while training is running")
                return False
            
            # Create load checkpoint signal file
            load_file = self.output_dir / "LOAD_CHECKPOINT"
            with open(load_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"dashboard_pid: {os.getpid()}\n")
                f.write(f"checkpoint_path: {checkpoint_path}\n")
            
            logger.info(f"Created load checkpoint signal file: {load_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get training status.
        
        Returns:
            Dict[str, Any]: Training status
        """
        try:
            # Check status file
            status_file = self.output_dir / "training_status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r") as f:
                        status = json.load(f)
                    
                    # Check if process is still running
                    if status.get("status") == "running" and status.get("pid"):
                        pid = status["pid"]
                        try:
                            os.kill(pid, 0)  # This will raise an exception if process is not running
                        except OSError:
                            # Process is not running
                            status["status"] = "stopped"
                            status["error"] = "Process not running"
                    
                    return status
                
                except Exception as e:
                    logger.error(f"Error reading training status: {e}")
            
            # Check for other status indicators
            if (self.output_dir / "PAUSE_TRAINING").exists():
                return {"status": "paused"}
            
            # Check for running process
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if (
                        "python" in proc.info["name"] and
                        proc.info["cmdline"] and
                        "training_orchestrator.py" in " ".join(proc.info["cmdline"]) and
                        self.experiment_name in " ".join(proc.info["cmdline"])
                    ):
                        return {
                            "status": "running",
                            "pid": proc.info["pid"],
                            "start_time": proc.create_time(),
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            # No status found
            return {"status": "unknown"}
        
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return {"status": "error", "error": str(e)}
    
    def execute_recovery_action(self, action: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute recovery action.
        
        Args:
            action: Recovery action to execute
            params: Parameters for the action
            
        Returns:
            bool: Whether the operation was successful
        """
        if self.read_only:
            logger.warning("Dashboard is in read-only mode, cannot execute recovery action")
            return False
        
        try:
            # Create recovery action signal file
            recovery_dir = self.output_dir / "recovery"
            recovery_dir.mkdir(exist_ok=True)
            
            action_file = recovery_dir / f"EXECUTE_{action.upper()}"
            with open(action_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"dashboard_pid: {os.getpid()}\n")
                f.write(f"action: {action}\n")
                
                if params:
                    for key, value in params.items():
                        f.write(f"{key}: {value}\n")
            
            logger.info(f"Created recovery action signal file: {action_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error executing recovery action: {e}")
            return False
    
    def send_notification(self, message: str, level: str = "info", channel: Optional[str] = None) -> bool:
        """
        Send notification.
        
        Args:
            message: Notification message
            level: Notification level (info, warning, error, critical)
            channel: Notification channel (slack, email, etc.)
            
        Returns:
            bool: Whether the operation was successful
        """
        try:
            # Check if notification is enabled
            if not self.config.notification_channels:
                logger.warning("Notifications are not enabled")
                return False
            
            # Determine channel
            if channel is None:
                channel = self.config.notification_channels[0]
            
            if channel not in self.config.notification_channels:
                logger.warning(f"Notification channel not enabled: {channel}")
                return False
            
            # Send notification
            if channel == "slack" and self.config.slack_webhook_url and REQUESTS_AVAILABLE:
                # Send Slack notification
                response = requests.post(
                    self.config.slack_webhook_url,
                    json={
                        "text": f"[{level.upper()}] {message}",
                        "username": f"Dashboard - {self.experiment_name}",
                    },
                timeout=5)
                
                return response.status_code == 200
            
            elif channel == "email" and self.config.email_recipients:
                # Send email notification
                # TODO: Implement email notification
                logger.warning("Email notifications not implemented yet")
                return False
            
            else:
                logger.warning(f"Unsupported notification channel: {channel}")
                return False
        
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False


class Dashboard:
    """Main dashboard application."""
    
    def __init__(self, config: DashboardConfig):
        """
        Initialize dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        self.experiment_data = ExperimentData(config)
        self.training_control = TrainingControl(config)
    
    def run(self):
        """Run the dashboard application."""
        if not STREAMLIT_AVAILABLE:
            logger.error("Streamlit not available. Install with: pip install streamlit")
            return
        
        # Set page config
        st.set_page_config(
            page_title=self.config.title,
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded" if self.config.dashboard_config["show_sidebar"] else "collapsed",
        )
        
        # Set theme
        if self.config.theme == "dark":
            st.markdown("""
                <style>
                    .reportview-container {
                        background-color: #111;
                        color: #fff;
                    }
                    .sidebar .sidebar-content {
                        background-color: #222;
                        color: #fff;
                    }
                </style>
            """, unsafe_allow_html=True)
        
        # Create sidebar
        self._create_sidebar()
        
        # Get current view
        view = st.session_state.get("view", self.config.default_view)
        
        # Render view
        if view == "overview":
            self._render_overview()
        elif view == "training":
            self._render_training()
        elif view == "system":
            self._render_system()
        elif view == "health":
            self._render_health()
        elif view == "failures":
            self._render_failures()
        elif view == "checkpoints":
            self._render_checkpoints()
        elif view == "ci":
            self._render_ci()
        elif view == "experiments":
            self._render_experiments()
        elif view == "costs":
            self._render_costs()
        elif view == "settings":
            self._render_settings()
        else:
            st.error(f"Unknown view: {view}")
    
    def _create_sidebar(self):
        """Create sidebar."""
        with st.sidebar:
            st.title(self.config.title)
            st.markdown(f"**Experiment:** {self.config.experiment_name}")
            
            # Add refresh button
            if st.button("Refresh Data"):
                self.experiment_data.update()
                st.success("Data refreshed")
            
            # Add auto-refresh checkbox
            auto_refresh = st.checkbox("Auto Refresh", value=self.config.auto_refresh)
            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh Interval (s)",
                    min_value=5,
                    max_value=60,
                    value=self.config.refresh_interval,
                    step=5,
                )
                
                # Set up auto-refresh
                if "last_refresh" not in st.session_state:
                    st.session_state.last_refresh = time.time()
                
                if time.time() - st.session_state.last_refresh > refresh_interval:
                    self.experiment_data.update()
                    st.session_state.last_refresh = time.time()
            
            # Add navigation
            st.markdown("## Navigation")
            
            if st.button("Overview", key="nav_overview"):
                st.session_state.view = "overview"
                st.experimental_rerun()
            
            if st.button("Training", key="nav_training"):
                st.session_state.view = "training"
                st.experimental_rerun()
            
            if st.button("System", key="nav_system"):
                st.session_state.view = "system"
                st.experimental_rerun()
            
            if st.button("Health", key="nav_health"):
                st.session_state.view = "health"
                st.experimental_rerun()
            
            if st.button("Failures", key="nav_failures"):
                st.session_state.view = "failures"
                st.experimental_rerun()
            
            if st.button("Checkpoints", key="nav_checkpoints"):
                st.session_state.view = "checkpoints"
                st.experimental_rerun()
            
            if st.button("CI/CD", key="nav_ci"):
                st.session_state.view = "ci"
                st.experimental_rerun()
            
            if st.button("Experiments", key="nav_experiments"):
                st.session_state.view = "experiments"
                st.experimental_rerun()
            
            if st.button("Costs", key="nav_costs"):
                st.session_state.view = "costs"
                st.experimental_rerun()
            
            if st.button("Settings", key="nav_settings"):
                st.session_state.view = "settings"
                st.experimental_rerun()
            
            # Add training controls
            if not self.config.read_only:
                st.markdown("## Training Controls")
                
                # Get training status
                status = self.training_control.get_training_status()
                st.markdown(f"**Status:** {status.get('status', 'unknown')}")
                
                # Add control buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if status.get("status") == "running":
                        if st.button("Stop", key="control_stop"):
                            if self.training_control.stop_training():
                                st.success("Training stopped")
                            else:
                                st.error("Failed to stop training")
                        
                        if st.button("Pause", key="control_pause"):
                            if self.training_control.pause_training():
                                st.success("Training paused")
                            else:
                                st.error("Failed to pause training")
                    
                    elif status.get("status") == "paused":
                        if st.button("Resume", key="control_resume"):
                            if self.training_control.resume_training():
                                st.success("Training resumed")
                            else:
                                st.error("Failed to resume training")
                    
                    else:
                        if st.button("Start", key="control_start"):
                            if self.training_control.start_training():
                                st.success("Training started")
                            else:
                                st.error("Failed to start training")
                
                with col2:
                    if st.button("Save Checkpoint", key="control_save"):
                        if self.training_control.save_checkpoint():
                            st.success("Checkpoint saved")
                        else:
                            st.error("Failed to save checkpoint")
            
            # Add export options
            st.markdown("## Export")
            
            export_type = st.selectbox(
                "Data Type",
                ["training", "system", "health", "failures", "checkpoints", "ci", "experiments", "all"],
            )
            
            export_format = st.selectbox(
                "Format",
                self.config.export_formats,
            )
            
            if st.button("Export", key="export"):
                path = self.experiment_data.export_data(export_type, export_format)
                if path:
                    st.success(f"Data exported to {path}")
                else:
                    st.error("Failed to export data")
            
            # Add footer
            st.markdown("---")
            st.markdown(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _render_overview(self):
        """Render overview page."""
        st.title("Training Overview")
        
        # Create layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Training status
            status = self.training_control.get_training_status()
            st.markdown(f"## Status: {status.get('status', 'unknown').capitalize()}")
            
            # Training metrics
            st.markdown("## Training Metrics")
            
            # Get key metrics
            metrics = self.experiment_data.get_training_metrics(self.config.default_metrics)
            
            if metrics:
                # Plot metrics
                if PLOTLY_AVAILABLE:
                    for source, source_metrics in metrics.items():
                        for metric_name, values in source_metrics.items():
                            if values:
                                steps, metric_values = zip(*values)
                                fig = px.line(
                                    x=steps,
                                    y=metric_values,
                                    title=f"{metric_name} ({source})",
                                    labels={"x": "Step", "y": "Value"},
                                )
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Plotly not available. Install with: pip install plotly")
                    st.json(metrics)
            else:
                st.info("No training metrics available")
            
            # Health status
            st.markdown("## Health Status")
            
            health = self.experiment_data.get_health_status()
            if health:
                if "heartbeat" in health:
                    heartbeat = health["heartbeat"]
                    st.markdown(f"**Heartbeat:** {heartbeat.get('status', 'unknown')}")
                    st.markdown(f"**Timestamp:** {heartbeat.get('timestamp', 'unknown')}")
                    
                    if "uptime" in heartbeat:
                        uptime = float(heartbeat["uptime"])
                        st.markdown(f"**Uptime:** {datetime.timedelta(seconds=int(uptime))}")
                
                if "latest" in health:
                    latest = health["latest"]
                    st.markdown(f"**Overall Health:** {latest.get('status', 'unknown')}")
                    
                    if "issues" in latest:
                        issues = latest["issues"]
                        if issues:
                            st.markdown("**Issues:**")
                            for issue in issues:
                                st.markdown(f"- {issue}")
                        else:
                            st.markdown("**Issues:** None")
            else:
                st.info("No health status available")
        
        with col2:
            # System resources
            st.markdown("## System Resources")
            
            system = self.experiment_data.get_system_metrics()
            if system and "current" in system:
                current = system["current"]
                
                # CPU
                if "cpu" in current:
                    cpu = current["cpu"]
                    st.markdown(f"**CPU:** {cpu.get('percent', 0):.1f}% ({cpu.get('count', 0)} cores)")
                
                # Memory
                if "memory" in current:
                    memory = current["memory"]
                    st.markdown(f"**Memory:** {memory.get('percent', 0):.1f}% ({memory.get('used', 0):.1f} GB / {memory.get('total', 0):.1f} GB)")
                
                # GPU
                if "gpu" in current and current["gpu"]:
                    st.markdown("**GPUs:**")
                    for gpu in current["gpu"]:
                        st.markdown(f"- {gpu.get('name', 'GPU')}: {gpu.get('memory_allocated', 0):.1f} GB allocated")
                
                # Disk
                if "disk" in current:
                    disk = current["disk"]
                    st.markdown(f"**Disk:** {disk.get('percent', 0):.1f}% ({disk.get('used', 0):.1f} GB / {disk.get('total', 0):.1f} GB)")
            else:
                st.info("No system resource data available")
            
            # Recent failures
            st.markdown("## Recent Failures")
            
            failures = self.experiment_data.get_failures()
            if failures and "data" in failures and "failures" in failures["data"]:
                failure_list = failures["data"]["failures"]
                if failure_list:
                    for i, failure in enumerate(failure_list[-5:]):  # Show last 5 failures
                        st.markdown(f"**{failure.get('failure_type', 'Unknown')}** at step {failure.get('step', 'N/A')}")
                        st.markdown(f"*{datetime.datetime.fromtimestamp(failure.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}*")
                        
                        if failure.get('recovery_success') is not None:
                            if failure.get('recovery_success'):
                                st.markdown("✅ Recovery successful")
                            else:
                                st.markdown("❌ Recovery failed")
                        
                        if i < len(failure_list[-5:]) - 1:
                            st.markdown("---")
                else:
                    st.markdown("No failures recorded")
            else:
                st.info("No failure data available")
            
            # Checkpoints
            st.markdown("## Latest Checkpoints")
            
            checkpoints = self.experiment_data.get_checkpoints()
            if checkpoints and "files" in checkpoints:
                checkpoint_list = checkpoints["files"]
                if checkpoint_list:
                    for i, checkpoint in enumerate(checkpoint_list[:3]):  # Show top 3 checkpoints
                        st.markdown(f"**{checkpoint.get('filename', 'Unknown')}**")
                        st.markdown(f"*{checkpoint.get('modified_date', 'Unknown')}*")
                        st.markdown(f"Size: {checkpoint.get('size_mb', 0):.1f} MB")
                        
                        if i < min(3, len(checkpoint_list)) - 1:
                            st.markdown("---")
                else:
                    st.markdown("No checkpoints available")
            else:
                st.info("No checkpoint data available")
            
            # CI/CD status
            st.markdown("## CI/CD Status")
            
            ci_jobs = self.experiment_data.get_ci_jobs()
            if ci_jobs and "latest" in ci_jobs:
                latest = ci_jobs["latest"]
                st.markdown(f"**Latest Job:** {latest.get('name', 'Unknown')}")
                st.markdown(f"**Status:** {latest.get('status', 'Unknown')}")
                st.markdown(f"**Started:** {latest.get('start_time', 'Unknown')}")
                
                if "conclusion" in latest:
                    st.markdown(f"**Conclusion:** {latest.get('conclusion', 'Unknown')}")
            else:
                st.info("No CI/CD data available")
    
    def _render_training(self):
        """Render training page."""
        st.title("Training Progress")
        
        # Create tabs
        tabs = st.tabs(["Metrics", "Learning Curves", "Progress", "Configuration"])
        
        with tabs[0]:
            # Training metrics
            st.markdown("## Training Metrics")
            
            # Get metrics
            metrics = self.experiment_data.get_training_metrics()
            
            if metrics:
                # Create metric selector
                all_metrics = []
                for source, source_metrics in metrics.items():
                    for metric_name in source_metrics.keys():
                        all_metrics.append(f"{source}:{metric_name}")
                
                selected_metrics = st.multiselect(
                    "Select Metrics",
                    all_metrics,
                    default=all_metrics[:min(5, len(all_metrics))],
                )
                
                # Plot selected metrics
                if selected_metrics and PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    
                    for metric in selected_metrics:
                        source, metric_name = metric.split(":", 1)
                        if source in metrics and metric_name in metrics[source]:
                            values = metrics[source][metric_name]
                            if values:
                                steps, metric_values = zip(*values)
                                fig.add_trace(go.Scatter(
                                    x=steps,
                                    y=metric_values,
                                    mode="lines",
                                    name=f"{metric