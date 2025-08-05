#!/usr/bin/env python
"""
Training Metrics Analysis Script for HRM-CodeGen

This script analyzes training metrics from W&B runs to detect anomalies,
identify patterns, and provide optimization recommendations. It generates
visualizations, compares runs across configurations, and produces detailed
reports on training performance.

Features:
- Anomaly detection in training metrics (loss, gradient norms, etc.)
- Performance trend visualization and statistical analysis
- Training efficiency and convergence analysis
- Comparative analysis across model configurations
- Optimization recommendations based on metrics analysis
- Integration with W&B for logging analysis results
- Comprehensive report generation with plots and statistics

Usage:
    python analyze_training_metrics.py --input recent_runs.json --output metrics_analysis.json
    python analyze_training_metrics.py --project hrm-codegen --days 7 --output metrics_analysis.json
    python analyze_training_metrics.py --input recent_runs.json --generate-plots --detect-anomalies
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("analyze_training_metrics")

# Try importing optional dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("matplotlib or seaborn not available, plotting will be disabled")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available, W&B integration will be disabled")

try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available, some statistical tests will be disabled")


class TrainingMetricsAnalyzer:
    """Analyzes training metrics from W&B runs."""

    def __init__(
        self,
        data: Union[str, Dict, pd.DataFrame],
        output_dir: str = "analysis_output",
        generate_plots: bool = True,
        detect_anomalies: bool = True,
        wandb_run_id: Optional[str] = None,
    ):
        """
        Initialize the training metrics analyzer.
        
        Args:
            data: Input data, either a path to a JSON file, a dictionary, or a DataFrame
            output_dir: Directory to save analysis outputs
            generate_plots: Whether to generate plots
            detect_anomalies: Whether to detect anomalies in metrics
            wandb_run_id: W&B run ID for logging analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)
        
        self.generate_plots = generate_plots and PLOTTING_AVAILABLE
        self.detect_anomalies = detect_anomalies
        self.wandb_run_id = wandb_run_id
        
        # Load data
        self.data = self._load_data(data)
        
        # Initialize results dictionary
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "metrics_analysis": {},
            "run_comparisons": {},
            "anomalies": {},
            "convergence_analysis": {},
            "efficiency_metrics": {},
            "recommendations": [],
            "statistical_tests": {},
        }
        
        # Set up W&B
        self._setup_wandb()
        
        # Configure plot style
        if self.generate_plots:
            sns.set(style="whitegrid")
            sns.set_palette("viridis")
            plt.rcParams["figure.figsize"] = (12, 8)
            plt.rcParams["figure.dpi"] = 100
    
    def _load_data(self, data: Union[str, Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Load data from various input formats.
        
        Args:
            data: Input data, either a path to a JSON file, a dictionary, or a DataFrame
            
        Returns:
            DataFrame containing the training metrics
        """
        if isinstance(data, str):
            # Load from JSON file
            logger.info(f"Loading data from {data}")
            with open(data, "r") as f:
                data_dict = json.load(f)
            
            # Convert to DataFrame
            if isinstance(data_dict, list):
                # List of runs
                return self._process_run_list(data_dict)
            else:
                # Single run or custom format
                return self._process_data_dict(data_dict)
        
        elif isinstance(data, dict):
            # Process dictionary
            return self._process_data_dict(data)
        
        elif isinstance(data, pd.DataFrame):
            # Already a DataFrame
            return data
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _process_run_list(self, runs: List[Dict]) -> pd.DataFrame:
        """
        Process a list of run dictionaries into a DataFrame.
        
        Args:
            runs: List of run dictionaries from W&B
            
        Returns:
            DataFrame containing the training metrics
        """
        all_dfs = []
        
        for run in runs:
            run_id = run.get("id", "unknown")
            run_name = run.get("name", run_id)
            config = run.get("config", {})
            
            # Extract history/metrics
            metrics = run.get("history", run.get("metrics", []))
            
            if not metrics:
                logger.warning(f"No metrics found for run {run_name} ({run_id})")
                continue
            
            # Convert to DataFrame
            run_df = pd.DataFrame(metrics)
            
            # Add run metadata
            run_df["run_id"] = run_id
            run_df["run_name"] = run_name
            
            # Add config as columns with 'config_' prefix
            for key, value in config.items():
                if isinstance(value, (int, float, str, bool)):
                    run_df[f"config_{key}"] = value
            
            all_dfs.append(run_df)
        
        if not all_dfs:
            raise ValueError("No valid run data found")
        
        # Combine all runs
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Processed {len(all_dfs)} runs with {len(combined_df)} data points")
        
        return combined_df
    
    def _process_data_dict(self, data_dict: Dict) -> pd.DataFrame:
        """
        Process a data dictionary into a DataFrame.
        
        Args:
            data_dict: Dictionary containing training metrics
            
        Returns:
            DataFrame containing the training metrics
        """
        # This is a flexible function that tries to handle various data formats
        if "runs" in data_dict:
            # Format with a 'runs' key containing a list of runs
            return self._process_run_list(data_dict["runs"])
        
        elif "metrics" in data_dict:
            # Format with a 'metrics' key
            metrics_df = pd.DataFrame(data_dict["metrics"])
            
            # Add metadata if available
            if "run_id" in data_dict:
                metrics_df["run_id"] = data_dict["run_id"]
            if "run_name" in data_dict:
                metrics_df["run_name"] = data_dict["run_name"]
            if "config" in data_dict:
                for key, value in data_dict["config"].items():
                    if isinstance(value, (int, float, str, bool)):
                        metrics_df[f"config_{key}"] = value
            
            return metrics_df
        
        else:
            # Try to convert the dictionary directly to a DataFrame
            try:
                return pd.DataFrame(data_dict)
            except ValueError:
                raise ValueError("Unsupported data dictionary format")
    
    def _setup_wandb(self):
        """Set up W&B for logging analysis results."""
        if WANDB_AVAILABLE and self.wandb_run_id:
            try:
                if wandb.run is None:
                    wandb.init(id=self.wandb_run_id, resume="allow")
                logger.info(f"W&B initialized with run ID: {self.wandb_run_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of training metrics.
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting training metrics analysis")
        
        # Ensure we have required columns
        required_metrics = ["loss"]
        missing_metrics = [m for m in required_metrics if m not in self.data.columns]
        if missing_metrics:
            logger.warning(f"Missing required metrics: {missing_metrics}")
        
        # Basic summary statistics
        self._analyze_summary_statistics()
        
        # Analyze individual metrics
        self._analyze_metrics()
        
        # Analyze convergence
        self._analyze_convergence()
        
        # Calculate efficiency metrics
        self._calculate_efficiency_metrics()
        
        # Compare runs
        self._compare_runs()
        
        # Detect anomalies if requested
        if self.detect_anomalies:
            self._detect_anomalies()
        
        # Perform statistical tests
        self._perform_statistical_tests()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Log results to W&B
        self._log_results_to_wandb()
        
        logger.info("Analysis completed successfully")
        
        return self.results
    
    def _analyze_summary_statistics(self):
        """Calculate summary statistics for all metrics."""
        logger.info("Calculating summary statistics")
        
        # Get numeric columns
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        
        # Remove run_id and step columns
        exclude_cols = ["run_id", "step", "_step", "_runtime", "_timestamp"]
        metric_cols = [col for col in numeric_cols if not any(col.startswith(ex) for ex in exclude_cols)]
        
        # Calculate statistics
        summary_stats = {}
        for metric in metric_cols:
            metric_data = self.data[metric].dropna()
            
            if len(metric_data) == 0:
                continue
                
            summary_stats[metric] = {
                "mean": float(metric_data.mean()),
                "median": float(metric_data.median()),
                "min": float(metric_data.min()),
                "max": float(metric_data.max()),
                "std": float(metric_data.std()),
                "q1": float(metric_data.quantile(0.25)),
                "q3": float(metric_data.quantile(0.75)),
                "count": int(len(metric_data)),
                "missing": int(self.data[metric].isna().sum()),
            }
        
        # Add run statistics
        run_counts = self.data["run_id"].value_counts()
        summary_stats["runs"] = {
            "total_runs": int(len(run_counts)),
            "total_steps": int(self.data["step"].max() if "step" in self.data.columns else 0),
            "avg_steps_per_run": float(self.data.groupby("run_id")["step"].max().mean() 
                                     if "step" in self.data.columns else 0),
        }
        
        self.results["summary"] = summary_stats
        
        # Generate summary plot if requested
        if self.generate_plots:
            self._plot_summary_statistics(metric_cols)
    
    def _plot_summary_statistics(self, metrics: List[str]):
        """
        Generate plots for summary statistics.
        
        Args:
            metrics: List of metric names to plot
        """
        # Filter to key metrics if there are too many
        if len(metrics) > 10:
            key_metrics = [m for m in ["loss", "val_loss", "learning_rate", "gradient_norm"] if m in metrics]
            if not key_metrics:
                key_metrics = metrics[:5]  # Take first 5 if no key metrics found
            metrics = key_metrics
        
        # Create distribution plots for each metric
        for metric in metrics:
            if metric not in self.data.columns:
                continue
                
            metric_data = self.data[metric].dropna()
            if len(metric_data) < 10:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Create a more informative plot combining histogram and KDE
            sns.histplot(metric_data, kde=True)
            
            plt.title(f"Distribution of {metric}")
            plt.xlabel(metric)
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            
            # Add vertical lines for key statistics
            plt.axvline(metric_data.mean(), color='r', linestyle='--', alpha=0.7, label=f"Mean: {metric_data.mean():.4f}")
            plt.axvline(metric_data.median(), color='g', linestyle='--', alpha=0.7, label=f"Median: {metric_data.median():.4f}")
            
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plot_dir / f"{metric}_distribution.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Log to W&B
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({f"analysis/distribution/{metric}": wandb.Image(str(plot_path))})
    
    def _analyze_metrics(self):
        """Analyze individual metrics and their trends."""
        logger.info("Analyzing individual metrics")
        
        # Get runs and their steps
        runs = self.data["run_id"].unique()
        
        # Get numeric columns (potential metrics)
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        
        # Remove run_id, config, and step columns
        exclude_prefixes = ["run_", "config_", "_"]
        exclude_cols = ["run_id", "step", "_step", "_runtime", "_timestamp"]
        metric_cols = [col for col in numeric_cols 
                      if col not in exclude_cols and 
                      not any(col.startswith(prefix) for prefix in exclude_prefixes)]
        
        # Analyze each metric
        metrics_analysis = {}
        
        for metric in metric_cols:
            if metric not in self.data.columns:
                continue
                
            metric_data = self.data[metric].dropna()
            if len(metric_data) < 10:
                continue
            
            # Calculate trend
            if "step" in self.data.columns:
                trend_data = self.data[["step", metric]].dropna()
                if len(trend_data) >= 10:
                    try:
                        # Use linear regression to estimate trend
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            trend_data["step"], trend_data[metric]
                        )
                        
                        trend_direction = "decreasing" if slope < 0 else "increasing"
                        trend_strength = abs(r_value)
                        
                        trend = {
                            "direction": trend_direction,
                            "slope": float(slope),
                            "r_squared": float(r_value ** 2),
                            "p_value": float(p_value),
                            "strength": "strong" if trend_strength > 0.7 else 
                                       "moderate" if trend_strength > 0.4 else "weak",
                        }
                    except Exception as e:
                        logger.warning(f"Error calculating trend for {metric}: {e}")
                        trend = {"error": str(e)}
                else:
                    trend = {"error": "insufficient data"}
            else:
                trend = {"error": "step column not available"}
            
            # Calculate volatility
            volatility = float(metric_data.std() / abs(metric_data.mean())) if abs(metric_data.mean()) > 0 else 0
            
            # Detect plateau
            plateau_detected = False
            plateau_at_step = None
            
            if "step" in self.data.columns and len(runs) > 0:
                # Check each run for plateau
                for run_id in runs:
                    run_data = self.data[self.data["run_id"] == run_id].sort_values("step")
                    if len(run_data) < 20:  # Need enough data to detect plateau
                        continue
                        
                    run_metric = run_data[metric].dropna()
                    if len(run_metric) < 20:
                        continue
                    
                    # Calculate rolling mean and std
                    window_size = min(20, len(run_metric) // 5)
                    rolling_mean = run_metric.rolling(window=window_size).mean()
                    rolling_std = run_metric.rolling(window=window_size).std()
                    
                    # Check if recent values have low std compared to overall
                    if len(rolling_std.dropna()) > 0:
                        recent_std = rolling_std.iloc[-10:].mean() if len(rolling_std) >= 10 else rolling_std.iloc[-1]
                        if recent_std < 0.01 * abs(run_metric.mean()):
                            plateau_detected = True
                            plateau_at_step = int(run_data["step"].iloc[-10])
                            break
            
            # Store analysis
            metrics_analysis[metric] = {
                "trend": trend,
                "volatility": volatility,
                "plateau": {
                    "detected": plateau_detected,
                    "at_step": plateau_at_step,
                },
            }
            
            # Generate plot if requested
            if self.generate_plots:
                self._plot_metric_analysis(metric)
        
        self.results["metrics_analysis"] = metrics_analysis
    
    def _plot_metric_analysis(self, metric: str):
        """
        Generate plots for metric analysis.
        
        Args:
            metric: Metric name to plot
        """
        if metric not in self.data.columns or "step" not in self.data.columns:
            return
            
        # Create a plot showing metric vs step for each run
        plt.figure(figsize=(12, 6))
        
        runs = self.data["run_id"].unique()
        
        for run_id in runs:
            run_data = self.data[self.data["run_id"] == run_id].sort_values("step")
            if "run_name" in run_data.columns:
                run_name = run_data["run_name"].iloc[0]
            else:
                run_name = run_id
                
            plt.plot(run_data["step"], run_data[metric], label=f"{run_name}", alpha=0.7)
        
        plt.title(f"{metric} vs Training Step")
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        
        # Add legend if there aren't too many runs
        if len(runs) <= 10:
            plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plot_dir / f"{metric}_vs_step.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log to W&B
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({f"analysis/metrics/{metric}_vs_step": wandb.Image(str(plot_path))})
        
        # Create a smoothed version with trend line for better visualization
        plt.figure(figsize=(12, 6))
        
        for run_id in runs:
            run_data = self.data[self.data["run_id"] == run_id].sort_values("step")
            if len(run_data) < 5:
                continue
                
            if "run_name" in run_data.columns:
                run_name = run_data["run_name"].iloc[0]
            else:
                run_name = run_id
            
            # Apply smoothing
            window_size = min(20, max(5, len(run_data) // 10))
            try:
                smoothed = run_data[metric].rolling(window=window_size, center=True).mean()
                plt.plot(run_data["step"], smoothed, label=f"{run_name} (smoothed)", alpha=0.7)
                
                # Add trend line
                if len(run_data) >= 10:
                    try:
                        # Fit line
                        z = np.polyfit(run_data["step"], run_data[metric], 1)
                        p = np.poly1d(z)
                        plt.plot(run_data["step"], p(run_data["step"]), 
                                 linestyle='--', alpha=0.5, 
                                 label=f"{run_name} trend (slope: {z[0]:.2e})")
                    except Exception:
                        pass
            except Exception:
                # Fall back to raw data if smoothing fails
                plt.plot(run_data["step"], run_data[metric], label=f"{run_name}", alpha=0.7)
        
        plt.title(f"{metric} vs Training Step (Smoothed with Trend)")
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        
        # Add legend if there aren't too many runs
        if len(runs) <= 5:
            plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plot_dir / f"{metric}_vs_step_smoothed.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log to W&B
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({f"analysis/metrics/{metric}_vs_step_smoothed": wandb.Image(str(plot_path))})
    
    def _analyze_convergence(self):
        """Analyze convergence patterns in training metrics."""
        logger.info("Analyzing convergence patterns")
        
        # Focus on loss metrics
        loss_metrics = [col for col in self.data.columns if "loss" in col.lower()]
        
        if not loss_metrics or "step" not in self.data.columns:
            logger.warning("Cannot analyze convergence: missing loss metrics or step column")
            return
        
        # Get runs
        runs = self.data["run_id"].unique()
        
        convergence_results = {}
        
        for metric in loss_metrics:
            metric_results = {}
            
            for run_id in runs:
                run_data = self.data[self.data["run_id"] == run_id].sort_values("step")
                if len(run_data) < 10:
                    continue
                    
                run_metric = run_data[metric].dropna()
                if len(run_metric) < 10:
                    continue
                
                # Get run name if available
                if "run_name" in run_data.columns:
                    run_name = run_data["run_name"].iloc[0]
                else:
                    run_name = run_id
                
                # Calculate time to convergence
                # (defined as when loss reaches within 10% of minimum)
                min_loss = run_metric.min()
                convergence_threshold = min_loss * 1.1
                
                steps_to_converge = None
                converged = False
                
                for i, (_, row) in enumerate(run_data.iterrows()):
                    if row[metric] <= convergence_threshold:
                        steps_to_converge = row["step"]
                        converged = True
                        break
                
                # Calculate convergence rate
                # (using exponential decay model: loss ~ initial_loss * exp(-rate * step))
                try:
                    # Get steps and corresponding loss values
                    steps = run_data["step"].values
                    losses = run_data[metric].values
                    
                    # Normalize steps to start from 0
                    steps = steps - steps[0]
                    
                    # Take log of losses
                    log_losses = np.log(losses + 1e-10)  # Add small epsilon to avoid log(0)
                    
                    # Fit linear model to log losses
                    slope, intercept, r_value, p_value, std_err = stats.linregress(steps, log_losses)
                    
                    # Convergence rate is the negative of the slope
                    convergence_rate = -slope
                    
                    # Half-life in steps (time to reduce loss by half)
                    if convergence_rate > 0:
                        half_life = np.log(2) / convergence_rate
                    else:
                        half_life = float('inf')
                    
                    # Goodness of fit
                    r_squared = r_value ** 2
                    
                    rate_analysis = {
                        "rate": float(convergence_rate),
                        "half_life_steps": float(half_life),
                        "r_squared": float(r_squared),
                        "model": "exponential_decay",
                    }
                except Exception as e:
                    logger.warning(f"Error calculating convergence rate for {run_name}: {e}")
                    rate_analysis = {"error": str(e)}
                
                # Detect oscillations
                oscillations_detected = False
                oscillation_amplitude = 0.0
                
                if len(run_metric) >= 20:
                    # Calculate differences between consecutive values
                    diffs = run_metric.diff().dropna()
                    
                    # Count sign changes (indicating oscillation)
                    sign_changes = ((diffs[:-1] * diffs[1:]) < 0).sum()
                    
                    # Calculate oscillation frequency
                    oscillation_frequency = sign_changes / len(diffs)
                    
                    # Detect significant oscillations
                    if oscillation_frequency > 0.3:  # More than 30% of points change direction
                        oscillations_detected = True
                        oscillation_amplitude = float(diffs.abs().mean())
                
                # Store results for this run
                metric_results[run_name] = {
                    "converged": converged,
                    "steps_to_converge": int(steps_to_converge) if steps_to_converge else None,
                    "min_value": float(min_loss),
                    "final_value": float(run_metric.iloc[-1]),
                    "convergence_rate": rate_analysis,
                    "oscillations": {
                        "detected": oscillations_detected,
                        "amplitude": float(oscillation_amplitude),
                    },
                }
                
                # Generate convergence plot if requested
                if self.generate_plots:
                    self._plot_convergence_analysis(run_data, metric, run_name, 
                                                   converged, steps_to_converge)
            
            convergence_results[metric] = metric_results
        
        self.results["convergence_analysis"] = convergence_results
    
    def _plot_convergence_analysis(self, run_data: pd.DataFrame, metric: str, 
                                  run_name: str, converged: bool, 
                                  steps_to_converge: Optional[int]):
        """
        Generate plots for convergence analysis.
        
        Args:
            run_data: DataFrame containing run data
            metric: Metric name to plot
            run_name: Name of the run
            converged: Whether the run converged
            steps_to_converge: Number of steps to convergence
        """
        plt.figure(figsize=(12, 6))
        
        # Plot the metric
        plt.plot(run_data["step"], run_data[metric], label=metric, alpha=0.7)
        
        # Add smoothed version
        window_size = min(20, max(5, len(run_data) // 10))
        try:
            smoothed = run_data[metric].rolling(window=window_size, center=True).mean()
            plt.plot(run_data["step"], smoothed, label=f"{metric} (smoothed)", 
                     linestyle='-', alpha=0.7)
        except Exception:
            pass
        
        # Mark convergence point if converged
        if converged and steps_to_converge is not None:
            convergence_idx = run_data[run_data["step"] == steps_to_converge].index[0]
            convergence_value = run_data.loc[convergence_idx, metric]
            
            plt.axvline(x=steps_to_converge, color='r', linestyle='--', alpha=0.5,
                       label=f"Converged at step {steps_to_converge}")
            plt.plot(steps_to_converge, convergence_value, 'ro', alpha=0.7)
        
        # Add min value line
        min_value = run_data[metric].min()
        plt.axhline(y=min_value, color='g', linestyle='--', alpha=0.5,
                   label=f"Min {metric}: {min_value:.4f}")
        
        plt.title(f"Convergence Analysis: {metric} for {run_name}")
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plot_dir / f"convergence_{run_name}_{metric}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log to W&B
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({f"analysis/convergence/{run_name}_{metric}": wandb.Image(str(plot_path))})
        
        # Create log-scale plot for better visualization of convergence
        plt.figure(figsize=(12, 6))
        
        # Plot the metric on log scale
        plt.semilogy(run_data["step"], run_data[metric], label=metric, alpha=0.7)
        
        # Mark convergence point if converged
        if converged and steps_to_converge is not None:
            convergence_idx = run_data[run_data["step"] == steps_to_converge].index[0]
            convergence_value = run_data.loc[convergence_idx, metric]
            
            plt.axvline(x=steps_to_converge, color='r', linestyle='--', alpha=0.5,
                       label=f"Converged at step {steps_to_converge}")
            plt.plot(steps_to_converge, convergence_value, 'ro', alpha=0.7)
        
        plt.title(f"Convergence Analysis (Log Scale): {metric} for {run_name}")
        plt.xlabel("Step")
        plt.ylabel(f"{metric} (log scale)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plot_dir / f"convergence_log_{run_name}_{metric}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log to W&B
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({f"analysis/convergence/log_{run_name}_{metric}": wandb.Image(str(plot_path))})
    
    def _calculate_efficiency_metrics(self):
        """Calculate training efficiency metrics."""
        logger.info("Calculating training efficiency metrics")
        
        # Check if we have the necessary columns
        if "step" not in self.data.columns or not any(col.lower().endswith("loss") for col in self.data.columns):
            logger.warning("Cannot calculate efficiency metrics: missing step or loss columns")
            return
        
        # Get runs
        runs = self.data["run_id"].unique()
        
        efficiency_metrics = {}
        
        for run_id in runs:
            run_data = self.data[self.data["run_id"] == run_id].sort_values("step")
            if len(run_data) < 10:
                continue
            
            # Get run name if available
            if "run_name" in run_data.columns:
                run_name = run_data["run_name"].iloc[0]
            else:
                run_name = run_id
            
            # Find primary loss metric (prefer 'loss' if available)
            loss_cols = [col for col in run_data.columns if "loss" in col.lower()]
            if "loss" in loss_cols:
                loss_metric = "loss"
            elif loss_cols:
                loss_metric = loss_cols[0]
            else:
                continue
            
            # Calculate efficiency metrics
            run_metrics = {}
            
            # 1. Loss improvement per step
            initial_loss = run_data[loss_metric].iloc[0]
            final_loss = run_data[loss_metric].iloc[-1]
            total_steps = run_data["step"].iloc[-1] - run_data["step"].iloc[0]
            
            if total_steps > 0:
                loss_improvement_per_step = (initial_loss - final_loss) / total_steps
            else:
                loss_improvement_per_step = 0
            
            # 2. Time to reach X% of final performance
            # (if we have runtime information)
            time_metrics = {}
            
            if "_runtime" in run_data.columns:
                initial_runtime = run_data["_runtime"].iloc[0]
                
                # Calculate time to reach various percentages of final loss
                for pct in [50, 75, 90, 95]:
                    target_loss = final_loss + (initial_loss - final_loss) * (1 - pct/100)
                    
                    # Find first step where loss is better than target
                    for i, (_, row) in enumerate(run_data.iterrows()):
                        if row[loss_metric] <= target_loss:
                            time_to_pct = row["_runtime"] - initial_runtime
                            time_metrics[f"time_to_{pct}pct"] = float(time_to_pct)
                            break
            
            # 3. Compute learning efficiency
            # (how quickly the model learns relative to computation used)
            if "_runtime" in run_data.columns:
                total_runtime = run_data["_runtime"].iloc[-1] - run_data["_runtime"].iloc[0]
                loss_improvement_per_second = (initial_loss - final_loss) / max(1, total_runtime)
            else:
                loss_improvement_per_second = None
            
            # 4. Sample efficiency
            # (if we have batch size information)
            if "config_global_batch_size" in run_data.columns:
                batch_size = run_data["config_global_batch_size"].iloc[0]
                total_samples = total_steps * batch_size
                loss_improvement_per_sample = (initial_loss - final_loss) / total_samples
            else:
                loss_improvement_per_sample = None
            
            # Store metrics
            run_metrics = {
                "loss_improvement_per_step": float(loss_improvement_per_step),
                "loss_improvement_per_second": float(loss_improvement_per_second) if loss_improvement_per_second is not None else None,
                "loss_improvement_per_sample": float(loss_improvement_per_sample) if loss_improvement_per_sample is not None else None,
                "time_metrics": time_metrics,
                "total_steps": int(total_steps),
                "initial_loss": float(initial_loss),
                "final_loss": float(final_loss),
                "loss_reduction_pct": float((initial_loss - final_loss) / initial_loss * 100) if initial_loss != 0 else 0,
            }
            
            efficiency_metrics[run_name] = run_metrics
        
        self.results["efficiency_metrics"] = efficiency_metrics
        
        # Generate efficiency comparison plot if requested
        if self.generate_plots and len(efficiency_metrics) > 1:
            self._plot_efficiency_comparison(efficiency_metrics)
    
    def _plot_efficiency_comparison(self, efficiency_metrics: Dict[str, Dict]):
        """
        Generate plots comparing efficiency metrics across runs.
        
        Args:
            efficiency_metrics: Dictionary of efficiency metrics by run
        """
        # Extract key metrics for comparison
        runs = list(efficiency_metrics.keys())
        loss_improvement_per_step = [efficiency_metrics[run]["loss_improvement_per_step"] for run in runs]
        loss_reduction_pct = [efficiency_metrics[run]["loss_reduction_pct"] for run in runs]
        
        # Create comparison plots
        plt.figure(figsize=(12, 6))
        
        # Plot loss improvement per step
        plt.subplot(1, 2, 1)
        bars = plt.bar(runs, loss_improvement_per_step, alpha=0.7)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2e}',
                    ha='center', va='bottom', rotation=0)
        
        plt.title("Loss Improvement per Step")
        plt.ylabel("Loss reduction per step")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)
        
        # Plot loss reduction percentage
        plt.subplot(1, 2, 2)
        bars = plt.bar(runs, loss_reduction_pct, alpha=0.7)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', rotation=0)
        
        plt.title("Total Loss Reduction")
        plt.ylabel("Loss reduction (%)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plot_dir / "efficiency_comparison.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log to W&B
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({"analysis/efficiency/comparison": wandb.Image(str(plot_path))})
        
        # Create time-to-percentage plot if we have that data
        has_time_metrics = any("time_metrics" in metrics and metrics["time_metrics"] 
                              for metrics in efficiency_metrics.values())
        
        if has_time_metrics:
            plt.figure(figsize=(10, 6))
            
            pct_levels = [50, 75, 90, 95]
            available_pcts = [pct for pct in pct_levels 
                             if any(f"time_to_{pct}pct" in metrics["time_metrics"] 
                                   for metrics in efficiency_metrics.values() 
                                   if "time_metrics" in metrics)]
            
            if not available_pcts:
                return
            
            # Prepare data
            plot_data = []
            for run in runs:
                if "time_metrics" not in efficiency_metrics[run]:
                    continue
                    
                time_metrics = efficiency_metrics[run]["time_metrics"]
                for pct in available_pcts:
                    key = f"time_to_{pct}pct"
                    if key in time_metrics:
                        plot_data.append({
                            "run": run,
                            "percentage": pct,
                            "time": time_metrics[key]
                        })
            
            if not plot_data:
                return
                
            # Convert to DataFrame for easier plotting
            plot_df = pd.DataFrame(plot_data)
            
            # Create grouped bar chart
            sns.barplot(x="percentage", y="time", hue="run", data=plot_df)
            
            plt.title("Time to Reach Performance Levels")
            plt.xlabel("Percentage of Final Performance")
            plt.ylabel("Time (seconds)")
            plt.grid(True, alpha=0.3)
            
            # Adjust legend if there are many runs
            if len(runs) > 5:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plot_dir / "time_to_performance.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Log to W&B
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({"analysis/efficiency/time_to_performance": wandb.Image(str(plot_path))})
    
    def _compare_runs(self):
        """Compare different runs and configurations."""
        logger.info("Comparing runs and configurations")
        
        # Get runs
        runs = self.data["run_id"].unique()
        
        if len(runs) <= 1:
            logger.info("Not enough runs to compare")
            return
        
        # Get run names if available
        run_names = {}
        for run_id in runs:
            run_data = self.data[self.data["run_id"] == run_id]
            if "run_name" in run_data.columns:
                run_names[run_id] = run_data["run_name"].iloc[0]
            else:
                run_names[run_id] = run_id
        
        # Extract configurations if available
        config_cols = [col for col in self.data.columns if col.startswith("config_")]
        
        if config_cols:
            # Get unique configurations
            configs = {}
            for run_id in runs:
                run_data = self.data[self.data["run_id"] == run_id]
                run_config = {col.replace("config_", ""): run_data[col].iloc[0] for col in config_cols 
                             if col in run_data.columns}
                configs[run_names[run_id]] = run_config
            
            # Find differences in configurations
            config_diffs = {}
            if len(configs) > 1:
                # Get all config keys
                all_keys = set()
                for config in configs.values():
                    all_keys.update(config.keys())
                
                # Find differences
                for key in all_keys:
                    values = {run: config.get(key, "N/A") for run, config in configs.items()}
                    if len(set(values.values())) > 1:
                        config_diffs[key] = values
            
            # Store configurations and differences
            self.results["run_comparisons"]["configurations"] = configs
            self.results["run_comparisons"]["config_differences"] = config_diffs
        
        # Compare performance metrics
        performance_metrics = {}
        
        # Focus on key metrics: loss and validation metrics
        key_metrics = [col for col in self.data.columns 
                      if "loss" in col.lower() or "val" in col.lower() or "accuracy" in col.lower()]
        
        if not key_metrics:
            # Fall back to any numeric columns that aren't config or metadata
            exclude_prefixes = ["run_", "config_", "_"]
            exclude_cols = ["run_id", "step", "_step", "_runtime", "_timestamp"]
            key_metrics = [col for col in self.data.select_dtypes(include=np.number).columns 
                          if col not in exclude_cols and 
                          not any(col.startswith(prefix) for prefix in exclude_prefixes)]
        
        # Compare final values for each metric
        for metric in key_metrics:
            metric_values = {}
            
            for run_id in runs:
                run_data = self.data[self.data["run_id"] == run_id].sort_values("step")
                if metric not in run_data.columns or len(run_data) == 0:
                    continue
                
                # Get final value
                final_value = run_data[metric].iloc[-1]
                metric_values[run_names[run_id]] = float(final_value)
            
            if metric_values:
                performance_metrics[metric] = metric_values
        
        # Determine best run for each metric
        best_runs = {}
        for metric, values in performance_metrics.items():
            if not values:
                continue
                
            # Determine if lower or higher is better
            lower_is_better = "loss" in metric.lower()
            
            if lower_is_better:
                best_run = min(values.items(), key=lambda x: x[1])
            else:
                best_run = max(values.items(), key=lambda x: x[1])
            
            best_runs[metric] = {
                "run": best_run[0],
                "value": best_run[1],
                "comparison": {run: (value - best_run[1]) / abs(best_run[1]) * 100 if best_run[1] != 0 else 0
                              for run, value in values.items()}
            }
        
        # Store performance comparison
        self.results["run_comparisons"]["performance_metrics"] = performance_metrics
        self.results["run_comparisons"]["best_runs"] = best_runs
        
        # Generate comparison plots if requested
        if self.generate_plots:
            self._plot_run_comparison(performance_metrics, best_runs)
    
    def _plot_run_comparison(self, performance_metrics: Dict[str, Dict], 
                            best_runs: Dict[str, Dict]):
        """
        Generate plots comparing performance across runs.
        
        Args:
            performance_metrics: Dictionary of performance metrics by run
            best_runs: Dictionary identifying the best run for each metric
        """
        # Create comparison plots for each metric
        for metric, values in performance_metrics.items():
            if not values or len(values) <= 1:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Determine if lower or higher is better
            lower_is_better = "loss" in metric.lower()
            
            # Sort runs by performance
            sorted_runs = sorted(values.items(), key=lambda x: x[1], 
                                reverse=not lower_is_better)
            
            runs = [item[0] for item in sorted_runs]
            metric_values = [item[1] for item in sorted_runs]
            
            # Create bar chart
            bars = plt.bar(runs, metric_values, alpha=0.7)
            
            # Highlight best run
            best_run = best_runs[metric]["run"]
            best_idx = runs.index(best_run)
            bars[best_idx].set_color('green')
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0)
            
            plt.title(f"Comparison of {metric} Across Runs")
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3)
            
            # Add note about which is better
            better_text = "Lower is better" if lower_is_better else "Higher is better"
            plt.annotate(better_text, xy=(0.02, 0.98), xycoords='axes fraction',
                        fontsize=10, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plot_dir / f"comparison_{metric}.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Log to W&B
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({f"analysis/comparison/{metric}": wandb.Image(str(plot_path))})
        
        # Create a summary plot showing relative performance across metrics
        if best_runs:
            # Prepare data for relative comparison
            metrics = list(best_runs.keys())
            
            # Get all runs
            all_runs = set()
            for metric, values in performance_metrics.items():
                all_runs.update(values.keys())
            all_runs = list(all_runs)
            
            # Create matrix of relative performances
            # (percentage difference from best run)
            relative_perf = np.zeros((len(all_runs), len(metrics)))
            
            for i, run in enumerate(all_runs):
                for j, metric in enumerate(metrics):
                    if run in best_runs[metric]["comparison"]:
                        # Get percentage difference from best
                        relative_perf[i, j] = best_runs[metric]["comparison"][run]
                    else:
                        relative_perf[i, j] = np.nan
            
            # Create heatmap
            plt.figure(figsize=(max(8, len(metrics) * 0.8), max(6, len(all_runs) * 0.5)))
            
            # Use a diverging colormap centered at 0
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
            
            # Create heatmap with custom formatting
            sns.heatmap(relative_perf, annot=True, fmt=".1f", 
                       xticklabels=metrics, yticklabels=all_runs,
                       cmap=cmap, center=0, vmin=-20, vmax=20,
                       cbar_kws={"label": "% difference from best"})
            
            plt.title("Relative Performance Comparison Across Metrics")
            plt.ylabel("Run")
            plt.xlabel("Metric")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plot_dir / "relative_performance_heatmap.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Log to W&B
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({"analysis/comparison/relative_performance": wandb.Image(str(plot_path))})
    
    def _detect_anomalies(self):
        """Detect anomalies in training metrics."""
        logger.info("Detecting anomalies in training metrics")
        
        # Get runs
        runs = self.data["run_id"].unique()
        
        # Get numeric columns (potential metrics)
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        
        # Remove run_id, config, and step columns
        exclude_prefixes = ["run_", "config_", "_"]
        exclude_cols = ["run_id", "step", "_step", "_runtime", "_timestamp"]
        metric_cols = [col for col in numeric_cols 
                      if col not in exclude_cols and 
                      not any(col.startswith(prefix) for prefix in exclude_prefixes)]
        
        # Detect anomalies for each run and metric
        anomalies = {}
        
        for run_id in runs:
            run_data = self.data[self.data["run_id"] == run_id].sort_values("step")
            if len(run_data) < 20:  # Need enough data for anomaly detection
                continue
                
            # Get run name if available
            if "run_name" in run_data.columns:
                run_name = run_data["run_name"].iloc[0]
            else:
                run_name = run_id
                
            run_anomalies = {}
            
            for metric in metric_cols:
                if metric not in run_data.columns:
                    continue
                    
                metric_data = run_data[metric].dropna()
                if len(metric_data) < 20:
                    continue
                
                # Detect anomalies using multiple methods
                anomaly_results = self._detect_metric_anomalies(run_data, metric)
                
                if anomaly_results["anomalies_detected"]:
                    run_anomalies[metric] = anomaly_results
                    
                    # Generate anomaly plot if requested
                    if self.generate_plots:
                        self._plot_anomalies(run_data, metric, run_name, anomaly_results)
            
            if run_anomalies:
                anomalies[run_name] = run_anomalies
        
        self.results["anomalies"] = anomalies
    
    def _detect_metric_anomalies(self, run_data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """
        Detect anomalies in a specific metric using multiple methods.
        
        Args:
            run_data: DataFrame containing run data
            metric: Metric name to analyze
            
        Returns:
            Dictionary containing anomaly detection results
        """
        # Extract metric data
        metric_data = run_data[metric].dropna().values
        
        if len(metric_data) < 20:
            return {"anomalies_detected": False, "error": "insufficient data"}
        
        # Prepare result dictionary
        result = {
            "anomalies_detected": False,
            "methods": {},
        }
        
        # Method 1: Statistical (Z-score)
        try:
            # Calculate z-scores
            z_scores = stats.zscore(metric_data)
            
            # Identify anomalies (|z| > 3)
            z_anomalies = np.abs(z_scores) > 3
            z_anomaly_indices = np.where(z_anomalies)[0]
            
            # Get corresponding steps
            if "step" in run_data.columns:
                z_anomaly_steps = run_data["step"].iloc[z_anomaly_indices].tolist()
            else:
                z_anomaly_steps = z_anomaly_indices.tolist()
            
            # Store results
            result["methods"]["z_score"] = {
                "anomalies_detected": len(z_anomaly_indices) > 0,
                "num_anomalies": int(len(z_anomaly_indices)),
                "anomaly_indices": z_anomaly_indices.tolist(),
                "anomaly_steps": z_anomaly_steps,
                "anomaly_values": metric_data[z_anomaly_indices].tolist(),
                "anomaly_z_scores": z_scores[z_anomaly_indices].tolist(),
            }
            
            if len(z_anomaly_indices) > 0:
                result["anomalies_detected"] = True
        except Exception as e:
            result["methods"]["z_score"] = {"error": str(e)}
        
        # Method 2: Isolation Forest
        try:
            # Reshape data for sklearn
            X = metric_data.reshape(-1, 1)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            iso_preds = iso_forest.fit_predict(X)
            
            # Identify anomalies (predicted as -1)
            iso_anomalies = iso_preds == -1
            iso_anomaly_indices = np.where(iso_anomalies)[0]
            
            # Get corresponding steps
            if "step" in run_data.columns:
                iso_anomaly_steps = run_data["step"].iloc[iso_anomaly_indices].tolist()
            else:
                iso_anomaly_steps = iso_anomaly_indices.tolist()
            
            # Store results
            result["methods"]["isolation_forest"] = {
                "anomalies_detected": len(iso_anomaly_indices) > 0,
                "num_anomalies": int(len(iso_anomaly_indices)),
                "anomaly_indices": iso_anomaly_indices.tolist(),
                "anomaly_steps": iso_anomaly_steps,
                "anomaly_values": metric_data[iso_anomaly_indices].tolist(),
            }
            
            if len(iso_anomaly_indices) > 0:
                result["anomalies_detected"] = True
        except Exception as e:
            result["methods"]["isolation_forest"] = {"error": str(e)}
        
        # Method 3: DBSCAN
        try:
            # Reshape data for sklearn
            X = metric_data.reshape(-1, 1)
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X_scaled)
            
            # Identify anomalies (labeled as -1)
            dbscan_anomalies = dbscan_labels == -1
            dbscan_anomaly_indices = np.where(dbscan_anomalies)[0]
            
            # Get corresponding steps
            if "step" in run_data.columns:
                dbscan_anomaly_steps = run_data["step"].iloc[dbscan_anomaly_indices].tolist()
            else:
                dbscan_anomaly_steps = dbscan_anomaly_indices.tolist()
            
            # Store results
            result["methods"]["dbscan"] = {
                "anomalies_detected": len(dbscan_anomaly_indices) > 0,
                "num_anomalies": int(len(dbscan_anomaly_indices)),
                "anomaly_indices": dbscan_anomaly_indices.tolist(),
                "anomaly_steps": dbscan_anomaly_steps,
                "anomaly_values": metric_data[dbscan_anomaly_indices].tolist(),
            }
            
            if len(dbscan_anomaly_indices) > 0:
                result["anomalies_detected"] = True
        except Exception as e:
            result["methods"]["dbscan"] = {"error": str(e)}
        
        # Method 4: Rolling window analysis
        try:
            # Calculate rolling statistics
            window_size = min(20, len(metric_data) // 5)
            rolling_mean = pd.Series(metric_data).rolling(window=window_size).mean()
            rolling_std = pd.Series(metric_data).rolling(window=window_size).std()
            
            # Calculate bounds
            upper_bound = rolling_mean + 3 * rolling_std
            lower_bound = rolling_mean - 3 * rolling_std
            
            # Identify anomalies
            window_anomalies = ((pd.Series(metric_data) > upper_bound) | 
                              (pd.Series(metric_data) < lower_bound))
            window_anomalies = window_anomalies.fillna(False)
            window_anomaly_indices = np.where(window_anomalies)[0]
            
            # Get corresponding steps
            if "step" in run_data.columns:
                window_anomaly_steps = run_data["step"].iloc[window_anomaly_indices].tolist()
            else:
                window_anomaly_steps = window_anomaly_indices.tolist()
            
            # Store results
            result["methods"]["rolling_window"] = {
                "anomalies_detected": len(window_anomaly_indices) > 0,
                "num_anomalies": int(len(window_anomaly_indices)),
                "anomaly_indices": window_anomaly_indices.tolist(),
                "anomaly_steps": window_anomaly_steps,
                "anomaly_values": metric_data[window_anomaly_indices].tolist(),
                "window_size": window_size,
            }
            
            if len(window_anomaly_indices) > 0:
                result["anomalies_detected"] = True
        except Exception as e:
            result["methods"]["rolling_window"] = {"error": str(e)}
        
        # Combine anomalies from all methods
        all_anomaly_indices = set()
        for method, method_results in result["methods"].items():
            if isinstance(method_results, dict) and "anomaly_indices" in method_results:
                all_anomaly_indices.update(method_results["anomaly_indices"])
        
        result["combined"] = {
            "anomalies_detected": len(all_anomaly_indices) > 0,
            "num_anomalies": len(all_anomaly_indices),
            "anomaly_indices": sorted(list(all_anomaly_indices)),
        }
        
        # Get corresponding steps and values
        if all_anomaly_indices:
            all_anomaly_indices = sorted(list(all_anomaly_indices))
            
            if "step" in run_data.columns:
                all_anomaly_steps = run_data["step"].iloc[all_anomaly_indices].tolist()
            else:
                all_anomaly_steps = all_anomaly_indices
                
            result["combined"]["anomaly_steps"] = all_anomaly_steps
            result["combined"]["anomaly_values"] = metric_data[all_anomaly_indices].tolist()
        
        return result
    
    def _plot_anomalies(self, run_data: pd.DataFrame, metric: str, 
                       run_name: str, anomaly_results: Dict[str, Any]):
        """
        Generate plots highlighting detected anomalies.
        
        Args:
            run_data: DataFrame containing run data
            metric: Metric name to plot
            run_name: Name of the run
            anomaly_results: Results from anomaly detection
        """
        if "step" not in run_data.columns:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot the metric
        plt.plot(run_data["step"], run_data[metric], label=metric, alpha=0.7)
        
        # Plot anomalies from each method
        for method, method_results in anomaly_results["methods"].items():
            if isinstance(method_results, dict) and "anomaly_steps" in method_results:
                anomaly_steps = method_results["anomaly_steps"]
                anomaly_values = method_results["anomaly_values"]
                
                if anomaly_steps and anomaly_values:
                    plt.scatter(anomaly_steps, anomaly_values, 
                               label=f"{method} anomalies ({len(anomaly_steps)})",
                               alpha=0.7, s=50)
        
        plt.title(f"Anomaly Detection: {metric} for {run_name}")
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plot_dir / f"anomalies_{run_name}_{metric}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log to W&B
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({f"analysis/anomalies/{run_name}_{metric}": wandb.Image(str(plot_path))})
    
    def _perform_statistical_tests(self):
        """Perform statistical tests on training metrics."""
        logger.info("Performing statistical tests")
        
        # Get runs
        runs = self.data["run_id"].unique()
        
        if len(runs) <= 1:
            logger.info("Not enough runs for statistical comparison")
            return
        
        # Get run names if available
        run_names = {}
        for run_id in runs:
            run_data = self.data[self.data["run_id"] == run_id]
            if "run_name" in run_data.columns:
                run_names[run_id] = run_data["run_name"].iloc[0]
            else:
                run_names[run_id] = run_id
        
        # Get numeric columns (potential metrics)
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        
        # Remove run_id, config, and step columns
        exclude_prefixes = ["run_", "config_", "_"]
        exclude_cols = ["run_id", "step", "_step", "_runtime", "_timestamp"]
        metric_cols = [col for col in numeric_cols 
                      if col not in exclude_cols and 
                      not any(col.startswith(prefix) for prefix in exclude_prefixes)]
        
        # Focus on key metrics
        key_metrics = [col for col in metric_cols 
                      if "loss" in col.lower() or "val" in col.lower() or "accuracy" in col.lower()]
        
        if not key_metrics:
            key_metrics = metric_cols[:5] if len(metric_cols) > 5 else metric_cols
        
        # Perform tests
        statistical_tests = {}
        
        # 1. Test for stationarity (Augmented Dickey-Fuller)
        if STATSMODELS_AVAILABLE:
            stationarity_tests = {}
            
            for run_id in runs:
                run_data = self.data[self.data["run_id"] == run_id].sort_values("step")
                run_name = run_names[run_id]
                
                run_tests = {}
                
                for metric in key_metrics:
                    if metric not in run_data.columns:
                        continue
                        
                    metric_data = run_data[metric].dropna()
                    if len(metric_data) < 20:
                        continue
                    
                    try:
                        # Perform ADF test
                        adf_result = adfuller(metric_data)
                        
                        # Interpret results
                        adf_statistic = adf_result[0]
                        p_value = adf_result[1]
                        is_stationary = p_value < 0.05
                        
                        run_tests[metric] = {
                            "adf_statistic": float(adf_statistic),
                            "p_value": float(p_value),
                            "is_stationary": is_stationary,
                            "critical_values": {str(key): float(val) for key, val in adf_result[4].items()},
                        }
                    except Exception as e:
                        run_tests[metric] = {"error": str(e)}
                
                if run_tests:
                    stationarity_tests[run_name] = run_tests
            
            if stationarity_tests:
                statistical_tests["stationarity"] = stationarity_tests
        
        # 2. Pairwise comparison of runs (t-test, Mann-Whitney)
        if len(runs) > 1:
            pairwise_tests = {}
            
            for metric in key_metrics:
                metric_tests = {}
                
                # Get data for each run
                run_data_dict = {}
                for run_id in runs:
                    run_data = self.data[self.data["run_id"] == run_id]
                    if metric in run_data.columns:
                        metric_data = run_data[metric].dropna()
                        if len(metric_data) >= 10:
                            run_data_dict[run_names[run_id]] = metric_data
                
                # Perform pairwise tests
                run_names_list = list(run_data_dict.keys())
                for i in range(len(run_names_list)):
                    for j in range(i+1, len(run_names_list)):
                        run1 = run_names_list[i]
                        run2 = run_names_list[j]
                        
                        data1 = run_data_dict[run1]
                        data2 = run_data_dict[run2]
                        
                        try:
                            # T-test
                            t_stat, t_p = stats.ttest_ind(data1, data2, equal_var=False)
                            
                            # Mann-Whitney U test
                            u_stat, u_p = stats.mannwhitneyu(data1, data2)
                            
                            # Store results
                            pair_name = f"{run1}_vs_{run2}"
                            metric_tests[pair_name] = {
                                "t_test": {
                                    "statistic": float(t_stat),
                                    "p_value": float(t_p),
                                    "significant": t_p < 0.05,
                                },
                                "mann_whitney": {
                                    "statistic": float(u_stat),
                                    "p_value": float(u_p),
                                    "significant": u_p < 0.05,
                                },
                                "mean_difference": float(data1.mean() - data2.mean()),
                                "percent_difference": float((data1.mean() - data2.mean()) / abs(data2.mean()) * 100) if data2.mean() != 0 else 0,
                            }
                        except Exception as e:
                            metric_tests[f"{run1}_vs_{run2}"] = {"error": str(e)}
                
                if metric_tests:
                    pairwise_tests[metric] = metric_tests
            
            if pairwise_tests:
                statistical_tests["pairwise_comparison"] = pairwise_tests
        
        # 3. Time series decomposition
        if STATSMODELS_AVAILABLE:
            decomposition_tests = {}
            
            for run_id in runs:
                run_data = self.data[self.data["run_id"] == run_id].sort_values("step")
                run_name = run_names[run_id]
                
                run_tests = {}
                
                for metric in key_metrics:
                    if metric not in run_data.columns:
                        continue
                        
                    metric_data = run_data[metric].dropna()
                    if len(metric_data) < 20:
                        continue
                    
                    try:
                        # Perform decomposition
                        # Need to ensure index is evenly spaced for seasonal_decompose
                        ts = pd.Series(metric_data.values, index=range(len(metric_data)))
                        
                        # Determine period (use 1/4 of the data length, but at least 4)
                        period = max(4, len(ts) // 4)
                        
                        # Decompose
                        result = seasonal_decompose(ts, model='additive', period=period)
                        
                        # Extract components
                        trend = result.trend.dropna()
                        seasonal = result.seasonal.dropna()
                        residual = result.resid.dropna()
                        
                        # Calculate component strengths
                        total_variance = np.var(ts)
                        trend_strength = np.var(trend) / total_variance if total_variance > 0 else 0
                        seasonal_strength = np.var(seasonal) / total_variance if total_variance > 0 else 0
                        residual_strength = np.var(residual) / total_variance if total_variance > 0 else 0
                        
                        run_tests[metric] = {
                            "trend_strength": float(trend_strength),
                            "seasonal_strength": float(seasonal_strength),
                            "residual_strength": float(residual_strength),
                            "period": period,
                        }
                        
                        # Generate decomposition plot if requested
                        if self.generate_plots:
                            self._plot_time_series_decomposition(result, metric, run_name)
                    except Exception as e:
                        run_tests[metric] = {"error": str(e)}
                
                if run_tests:
                    decomposition_tests[run_name] = run_tests
            
            if decomposition_tests:
                statistical_tests["decomposition"] = decomposition_tests
        
        self.results["statistical_tests"] = statistical_tests
    
    def _plot_time_series_decomposition(self, decomposition_result, metric: str, run_name: str):
        """
        Generate plot for time series decomposition.
        
        Args:
            decomposition_result: Result from seasonal_decompose
            metric: Metric name
            run_name: Name of the run
        """
        plt.figure(figsize=(12, 10))
        
        # Plot decomposition components
        plt.subplot(411)
        plt.plot(decomposition_result.observed)
        plt.title(f"Time Series Decomposition: {metric} for {run_name}")
        plt.ylabel("Observed")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(412)
        plt.plot(decomposition_result.trend)
        plt.ylabel("Trend")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(413)
        plt.plot(decomposition_result.seasonal)
        plt.ylabel("Seasonal")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(414)
        plt.plot(decomposition_result.resid)
        plt.ylabel("Residual")
        plt.xlabel("Time")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plot_dir / f"decomposition_{run_name}_{metric}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log to W&B
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({f"analysis/decomposition/{run_name}_{metric}": wandb.Image(str(plot_path))})
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis results."""
        logger.info("Generating recommendations")
        
        recommendations = []
        
        # 1. Learning rate recommendations
        if "metrics_analysis" in self.results:
            # Check if loss is plateauing
            for metric, analysis in self.results["metrics_analysis"].items():
                if "loss" in metric.lower() and "plateau" in analysis:
                    if analysis["plateau"]["detected"]:
                        recommendations.append({
                            "category": "learning_rate",
                            "issue": f"{metric} has plateaued",
                            "recommendation": "Consider increasing the learning rate or using a learning rate scheduler with warm restarts.",
                            "confidence": "medium",
                            "related_metrics": [metric],
                        })
        
        # 2. Batch size recommendations
        if "efficiency_metrics" in self.results:
            efficiency_metrics = self.results["efficiency_metrics"]
            
            # Check if any runs have very low efficiency
            low_efficiency_runs = []
            for run, metrics in efficiency_metrics.items():
                if "loss_improvement_per_step" in metrics and metrics["loss_improvement_per_step"] < 1e-5:
                    low_efficiency_runs.append(run)
            
            if low_efficiency_runs:
                recommendations.append({
                    "category": "batch_size",
                    "issue": f"Low training efficiency in runs: {', '.join(low_efficiency_runs)}",
                    "recommendation": "Consider increasing the batch size to improve training efficiency, or decrease it if you're experiencing poor convergence.",
                    "confidence": "medium",
                    "related_runs": low_efficiency_runs,
                })
        
        # 3. Convergence recommendations
        if "convergence_analysis" in self.results:
            for metric, runs in self.results["convergence_analysis"].items():
                # Check for oscillations
                oscillating_runs = []
                for run, analysis in runs.items():
                    if "oscillations" in analysis and analysis["oscillations"]["detected"]:
                        oscillating_runs.append(run)
                
                if oscillating_runs:
                    recommendations.append({
                        "category": "optimizer",
                        "issue": f"Oscillations detected in {metric} for runs: {', '.join(oscillating_runs)}",
                        "recommendation": "Consider reducing the learning rate or using an optimizer with momentum to dampen oscillations.",
                        "confidence": "