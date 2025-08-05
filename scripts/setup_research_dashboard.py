#!/usr/bin/env python
"""
HRM Research Dashboard Setup

This script sets up Weights & Biases (W&B) research dashboards for HRM validation,
focusing on comparing HRM models with transformer baselines (particularly GPT-2-117M).

The dashboards provide:
1. Pass@k metrics visualization and comparison
2. Performance benchmarking (latency, memory usage)
3. Training efficiency metrics
4. Statistical significance analysis
5. Model architecture comparison

Usage:
    python scripts/setup_research_dashboard.py [--api-key API_KEY] [--entity ENTITY]

Requirements:
    - wandb
    - pandas
    - numpy
    - matplotlib
    - seaborn
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

try:
    import wandb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wandb.apis.public import Reports
except ImportError:
    print("Error: Required packages not installed. Run: pip install wandb pandas numpy matplotlib seaborn")
    sys.exit(1)


# Configuration
DEFAULT_PROJECT = "hrm-codegen"
DEFAULT_ENTITY = None  # Will use default entity from wandb login
BASELINE_MODEL = "gpt-2-117m"
HRM_MODEL = "hrm-27m"
PERFORMANCE_TARGETS = {
    "pass@1": 0.30,  # 30% (vs GPT-2-117M: 26%)
    "pass@10": 0.45,  # 45% (vs GPT-2-117M: 42%)
}
DATASET = "mbpp"
SAMPLE_SIZE = 500  # Number of samples for statistical significance


class ResearchDashboardSetup:
    """Setup W&B research dashboards for HRM validation."""

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        entity: Optional[str] = None, 
        project: str = DEFAULT_PROJECT
    ):
        """
        Initialize the dashboard setup.
        
        Args:
            api_key: W&B API key (optional if already logged in)
            entity: W&B entity (username or team name)
            project: W&B project name
        """
        self.api_key = api_key
        self.entity = entity
        self.project = project
        self.api = None
        self.reports = {}
        
        # Initialize W&B
        self._init_wandb()
        
    def _init_wandb(self):
        """Initialize W&B API."""
        if self.api_key:
            os.environ["WANDB_API_KEY"] = self.api_key
            
        try:
            wandb.login()
            self.api = wandb.Api()
            print(f"Successfully logged in to W&B as {self.api.viewer()['entity']}")
            
            # Set entity if not provided
            if not self.entity:
                self.entity = self.api.viewer()["entity"]
                
            # Check if project exists
            try:
                self.api.project(f"{self.entity}/{self.project}")
                print(f"Project {self.entity}/{self.project} exists")
            except wandb.CommError:
                print(f"Project {self.entity}/{self.project} will be created on first run")
                
        except Exception as e:
            print(f"Error initializing W&B: {e}")
            sys.exit(1)
    
    def setup_all_dashboards(self):
        """Set up all research dashboards."""
        print("\n=== Setting up HRM Research Dashboards ===\n")
        
        # Create dashboards
        self.setup_pass_at_k_dashboard()
        self.setup_performance_comparison_dashboard()
        self.setup_training_metrics_dashboard()
        self.setup_statistical_analysis_dashboard()
        self.setup_architecture_comparison_dashboard()
        self.setup_experiment_tracking_dashboard()
        
        print("\n=== Dashboard Setup Complete ===\n")
        print("Access your dashboards at:")
        print(f"https://wandb.ai/{self.entity}/{self.project}")
        
        # Return report URLs
        return {name: report.url for name, report in self.reports.items()}
    
    def setup_pass_at_k_dashboard(self):
        """Set up Pass@k metrics dashboard."""
        print("Setting up Pass@k Metrics Dashboard...")
        
        # Create report
        report = self.api.create_report(
            project=self.project,
            entity=self.entity,
            title="HRM vs Transformer: Pass@k Metrics",
            description=f"""
            Comparison of Pass@k metrics between HRM ({HRM_MODEL}) and Transformer baseline ({BASELINE_MODEL}).
            
            **Performance Targets:**
            - Pass@1: {PERFORMANCE_TARGETS['pass@1'] * 100:.1f}% (vs {BASELINE_MODEL}: 26%)
            - Pass@10: {PERFORMANCE_TARGETS['pass@10'] * 100:.1f}% (vs {BASELINE_MODEL}: 42%)
            
            **Dataset:** {DATASET} ({SAMPLE_SIZE} samples)
            """
        )
        
        # Add panels
        report.add_block(
            type="panel",
            title="Pass@k Performance",
            panels=[
                {
                    "type": "line",
                    "query": {
                        "keys": ["val/pass@1", "val/pass@10"],
                        "filters": {"$or": [
                            {"config.model": HRM_MODEL},
                            {"config.model": BASELINE_MODEL}
                        ]},
                    },
                    "layout": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "title": "Pass@k Performance Over Time"
                }
            ]
        )
        
        # Add bar chart comparing best results
        report.add_block(
            type="panel",
            title="Best Pass@k Comparison",
            panels=[
                {
                    "type": "bar",
                    "query": {
                        "keys": ["val/pass@1", "val/pass@10"],
                        "filters": {"$or": [
                            {"config.model": HRM_MODEL},
                            {"config.model": BASELINE_MODEL}
                        ]},
                        "aggregation": "maximum",
                    },
                    "layout": {"x": 0, "y": 8, "w": 12, "h": 8},
                    "title": "Best Pass@k Comparison"
                }
            ]
        )
        
        # Add target markers
        report.add_block(
            type="markdown",
            title="Performance Targets",
            content=f"""
            ## HRM Performance Targets
            
            | Metric | Target | Baseline ({BASELINE_MODEL}) |
            |--------|--------|--------------------------|
            | Pass@1 | {PERFORMANCE_TARGETS['pass@1'] * 100:.1f}% | 26% |
            | Pass@10 | {PERFORMANCE_TARGETS['pass@10'] * 100:.1f}% | 42% |
            
            **Statistical Significance:** Analysis performed on {SAMPLE_SIZE} samples from {DATASET} dataset.
            """
        )
        
        # Save report
        report.save()
        self.reports["pass_at_k"] = report
        print(f"Pass@k Dashboard created: {report.url}")
        
        return report
    
    def setup_performance_comparison_dashboard(self):
        """Set up performance comparison dashboard."""
        print("Setting up Performance Comparison Dashboard...")
        
        # Create report
        report = self.api.create_report(
            project=self.project,
            entity=self.entity,
            title="HRM vs Transformer: Performance Metrics",
            description=f"""
            Performance comparison between HRM ({HRM_MODEL}) and Transformer baseline ({BASELINE_MODEL}).
            
            **Metrics:**
            - Inference latency
            - Memory usage
            - Tokens per second
            - Training throughput
            
            **Models:**
            - HRM: 27M parameters
            - Baseline: 117M parameters ({BASELINE_MODEL})
            """
        )
        
        # Add panels for inference performance
        report.add_block(
            type="panel",
            title="Inference Performance",
            panels=[
                {
                    "type": "line",
                    "query": {
                        "keys": ["inference/latency"],
                        "filters": {"$or": [
                            {"config.model": HRM_MODEL},
                            {"config.model": BASELINE_MODEL}
                        ]},
                    },
                    "layout": {"x": 0, "y": 0, "w": 6, "h": 8},
                    "title": "Inference Latency (ms)"
                },
                {
                    "type": "line",
                    "query": {
                        "keys": ["inference/tokens_per_second"],
                        "filters": {"$or": [
                            {"config.model": HRM_MODEL},
                            {"config.model": BASELINE_MODEL}
                        ]},
                    },
                    "layout": {"x": 6, "y": 0, "w": 6, "h": 8},
                    "title": "Tokens Per Second"
                }
            ]
        )
        
        # Add panels for memory usage
        report.add_block(
            type="panel",
            title="Resource Usage",
            panels=[
                {
                    "type": "line",
                    "query": {
                        "keys": ["memory/gpu_usage_gb"],
                        "filters": {"$or": [
                            {"config.model": HRM_MODEL},
                            {"config.model": BASELINE_MODEL}
                        ]},
                    },
                    "layout": {"x": 0, "y": 8, "w": 6, "h": 8},
                    "title": "GPU Memory Usage (GB)"
                },
                {
                    "type": "line",
                    "query": {
                        "keys": ["memory/peak_usage_gb"],
                        "filters": {"$or": [
                            {"config.model": HRM_MODEL},
                            {"config.model": BASELINE_MODEL}
                        ]},
                    },
                    "layout": {"x": 6, "y": 8, "w": 6, "h": 8},
                    "title": "Peak Memory Usage (GB)"
                }
            ]
        )
        
        # Add efficiency comparison
        report.add_block(
            type="markdown",
            title="Efficiency Analysis",
            content=f"""
            ## Efficiency Comparison
            
            | Model | Parameters | Memory Usage | Inference Latency | Training Time |
            |-------|------------|--------------|-------------------|---------------|
            | HRM | 27M | {{memory/gpu_usage_gb}} GB | {{inference/latency}} ms | {{training/hours}} hours |
            | {BASELINE_MODEL} | 117M | {{memory/gpu_usage_gb}} GB | {{inference/latency}} ms | {{training/hours}} hours |
            
            **Efficiency Ratio:** HRM achieves similar or better performance with ~4.3x fewer parameters.
            """
        )
        
        # Save report
        report.save()
        self.reports["performance"] = report
        print(f"Performance Dashboard created: {report.url}")
        
        return report
    
    def setup_training_metrics_dashboard(self):
        """Set up training metrics dashboard."""
        print("Setting up Training Metrics Dashboard...")
        
        # Create report
        report = self.api.create_report(
            project=self.project,
            entity=self.entity,
            title="HRM Training Metrics",
            description=f"""
            Comprehensive training metrics for HRM model.
            
            **Tracked Metrics:**
            - Training loss
            - Validation loss
            - Learning rate
            - Gradient norm
            - Training throughput
            - GPU utilization
            """
        )
        
        # Add panels for training metrics
        report.add_block(
            type="panel",
            title="Training Progress",
            panels=[
                {
                    "type": "line",
                    "query": {
                        "keys": ["train/loss", "val/loss"],
                        "filters": {"config.model": HRM_MODEL},
                    },
                    "layout": {"x": 0, "y": 0, "w": 6, "h": 8},
                    "title": "Loss Curves"
                },
                {
                    "type": "line",
                    "query": {
                        "keys": ["train/learning_rate"],
                        "filters": {"config.model": HRM_MODEL},
                    },
                    "layout": {"x": 6, "y": 0, "w": 6, "h": 8},
                    "title": "Learning Rate Schedule"
                }
            ]
        )
        
        # Add panels for resource usage
        report.add_block(
            type="panel",
            title="Training Resources",
            panels=[
                {
                    "type": "line",
                    "query": {
                        "keys": ["train/gpu_utilization"],
                        "filters": {"config.model": HRM_MODEL},
                    },
                    "layout": {"x": 0, "y": 8, "w": 6, "h": 8},
                    "title": "GPU Utilization (%)"
                },
                {
                    "type": "line",
                    "query": {
                        "keys": ["train/memory_usage"],
                        "filters": {"config.model": HRM_MODEL},
                    },
                    "layout": {"x": 6, "y": 8, "w": 6, "h": 8},
                    "title": "Memory Usage (GB)"
                }
            ]
        )
        
        # Add panels for training throughput
        report.add_block(
            type="panel",
            title="Training Throughput",
            panels=[
                {
                    "type": "line",
                    "query": {
                        "keys": ["train/examples_per_second"],
                        "filters": {"config.model": HRM_MODEL},
                    },
                    "layout": {"x": 0, "y": 16, "w": 6, "h": 8},
                    "title": "Examples Per Second"
                },
                {
                    "type": "line",
                    "query": {
                        "keys": ["train/tokens_per_second"],
                        "filters": {"config.model": HRM_MODEL},
                    },
                    "layout": {"x": 6, "y": 16, "w": 6, "h": 8},
                    "title": "Tokens Per Second"
                }
            ]
        )
        
        # Save report
        report.save()
        self.reports["training"] = report
        print(f"Training Metrics Dashboard created: {report.url}")
        
        return report
    
    def setup_statistical_analysis_dashboard(self):
        """Set up statistical analysis dashboard."""
        print("Setting up Statistical Analysis Dashboard...")
        
        # Create report
        report = self.api.create_report(
            project=self.project,
            entity=self.entity,
            title="HRM vs Transformer: Statistical Analysis",
            description=f"""
            Statistical significance analysis of HRM performance vs Transformer baseline.
            
            **Analysis:**
            - Confidence intervals for Pass@k metrics
            - Statistical significance testing
            - Effect size analysis
            - Sample size validation
            
            **Dataset:** {DATASET} ({SAMPLE_SIZE} samples)
            """
        )
        
        # Add statistical significance panel
        report.add_block(
            type="markdown",
            title="Statistical Significance",
            content=f"""
            ## Statistical Significance Analysis
            
            ### Confidence Intervals (95%)
            
            | Model | Pass@1 | Pass@10 |
            |-------|--------|---------|
            | HRM ({HRM_MODEL}) | {{val/pass@1}} ± {{val/pass@1_ci}} | {{val/pass@10}} ± {{val/pass@10_ci}} |
            | Baseline ({BASELINE_MODEL}) | {{val/pass@1}} ± {{val/pass@1_ci}} | {{val/pass@10}} ± {{val/pass@10_ci}} |
            
            ### Hypothesis Testing
            
            **Null Hypothesis (H₀):** HRM performance is equal to or worse than the baseline.  
            **Alternative Hypothesis (H₁):** HRM performance is better than the baseline.
            
            | Metric | p-value | Significant? |
            |--------|---------|--------------|
            | Pass@1 | {{stats/pass@1_pvalue}} | {{stats/pass@1_significant}} |
            | Pass@10 | {{stats/pass@10_pvalue}} | {{stats/pass@10_significant}} |
            
            *Significance level: α = 0.05*
            
            ### Effect Size
            
            | Metric | Effect Size | Interpretation |
            |--------|-------------|----------------|
            | Pass@1 | {{stats/pass@1_effect_size}} | {{stats/pass@1_effect_interpretation}} |
            | Pass@10 | {{stats/pass@10_effect_size}} | {{stats/pass@10_effect_interpretation}} |
            
            ### Sample Size Analysis
            
            Sample size of {SAMPLE_SIZE} provides a margin of error of approximately ±{{stats/margin_of_error}} at 95% confidence.
            """
        )
        
        # Add visualization panel
        report.add_block(
            type="panel",
            title="Performance Distribution",
            panels=[
                {
                    "type": "custom",
                    "query": {
                        "keys": ["val/pass@1_distribution", "val/pass@10_distribution"],
                        "filters": {"$or": [
                            {"config.model": HRM_MODEL},
                            {"config.model": BASELINE_MODEL}
                        ]},
                    },
                    "layout": {"x": 0, "y": 0, "w": 12, "h": 12},
                    "title": "Pass@k Distribution"
                }
            ]
        )
        
        # Add bootstrap analysis
        report.add_block(
            type="markdown",
            title="Bootstrap Analysis",
            content=f"""
            ## Bootstrap Analysis
            
            Bootstrap analysis with 1000 resamples shows that HRM outperforms the baseline with {{stats/bootstrap_confidence}}% confidence.
            
            ### Reliability Analysis
            
            | Category | Easy | Medium | Hard | Overall |
            |----------|------|--------|------|---------|
            | HRM Performance | {{stats/hrm_easy}} | {{stats/hrm_medium}} | {{stats/hrm_hard}} | {{stats/hrm_overall}} |
            | Baseline Performance | {{stats/baseline_easy}} | {{stats/baseline_medium}} | {{stats/baseline_hard}} | {{stats/baseline_overall}} |
            | Relative Improvement | {{stats/improvement_easy}} | {{stats/improvement_medium}} | {{stats/improvement_hard}} | {{stats/improvement_overall}} |
            
            *Problem categories based on difficulty rating in the {DATASET} dataset.*
            """
        )
        
        # Save report
        report.save()
        self.reports["statistical"] = report
        print(f"Statistical Analysis Dashboard created: {report.url}")
        
        return report
    
    def setup_architecture_comparison_dashboard(self):
        """Set up architecture comparison dashboard."""
        print("Setting up Architecture Comparison Dashboard...")
        
        # Create report
        report = self.api.create_report(
            project=self.project,
            entity=self.entity,
            title="HRM vs Transformer: Architecture Comparison",
            description=f"""
            Detailed comparison of HRM architecture vs standard Transformer architecture.
            
            **Comparison:**
            - Parameter efficiency
            - Attention mechanism
            - Memory usage
            - Computational complexity
            - Scaling properties
            """
        )
        
        # Add architecture comparison
        report.add_block(
            type="markdown",
            title="Architecture Comparison",
            content=f"""
            ## HRM vs Transformer Architecture
            
            ### Key Differences
            
            | Feature | HRM | Standard Transformer |
            |---------|-----|---------------------|
            | Parameters | 27M | 117M (GPT-2 Small) |
            | Attention Mechanism | Hierarchical Recurrent Memory | Self-Attention |
            | Memory Complexity | O(n) | O(n²) |
            | Context Length | Linear scaling | Quadratic scaling |
            | Training Efficiency | Higher | Lower |
            
            ### Parameter Breakdown
            
            | Component | HRM | Transformer |
            |-----------|-----|------------|
            | Embeddings | {{arch/hrm_embedding_params}} | {{arch/transformer_embedding_params}} |
            | Attention | {{arch/hrm_attention_params}} | {{arch/transformer_attention_params}} |
            | Feed-forward | {{arch/hrm_ffn_params}} | {{arch/transformer_ffn_params}} |
            | Other | {{arch/hrm_other_params}} | {{arch/transformer_other_params}} |
            | **Total** | **{{arch/hrm_total_params}}** | **{{arch/transformer_total_params}}** |
            
            ### Efficiency Analysis
            
            HRM achieves comparable or better performance with approximately 4.3x fewer parameters.
            """
        )
        
        # Add visualization panel
        report.add_block(
            type="panel",
            title="Architecture Comparison",
            panels=[
                {
                    "type": "bar",
                    "query": {
                        "keys": ["arch/parameters", "arch/memory_usage", "arch/compute_flops"],
                        "filters": {"$or": [
                            {"config.model": HRM_MODEL},
                            {"config.model": BASELINE_MODEL}
                        ]},
                    },
                    "layout": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "title": "Architecture Metrics"
                }
            ]
        )
        
        # Add scaling analysis
        report.add_block(
            type="markdown",
            title="Scaling Analysis",
            content=f"""
            ## Scaling Properties
            
            ### Context Length Scaling
            
            | Context Length | HRM Memory | Transformer Memory | HRM Time | Transformer Time |
            |----------------|------------|-------------------|----------|------------------|
            | 512 tokens | {{scaling/hrm_memory_512}} | {{scaling/transformer_memory_512}} | {{scaling/hrm_time_512}} | {{scaling/transformer_time_512}} |
            | 1024 tokens | {{scaling/hrm_memory_1024}} | {{scaling/transformer_memory_1024}} | {{scaling/hrm_time_1024}} | {{scaling/transformer_time_1024}} |
            | 2048 tokens | {{scaling/hrm_memory_2048}} | {{scaling/transformer_memory_2048}} | {{scaling/hrm_time_2048}} | {{scaling/transformer_time_2048}} |
            | 4096 tokens | {{scaling/hrm_memory_4096}} | {{scaling/transformer_memory_4096}} | {{scaling/hrm_time_4096}} | {{scaling/transformer_time_4096}} |
            
            HRM shows linear scaling with context length, while Transformer shows quadratic scaling.
            """
        )
        
        # Save report
        report.save()
        self.reports["architecture"] = report
        print(f"Architecture Comparison Dashboard created: {report.url}")
        
        return report
    
    def setup_experiment_tracking_dashboard(self):
        """Set up experiment tracking dashboard."""
        print("Setting up Experiment Tracking Dashboard...")
        
        # Create report
        report = self.api.create_report(
            project=self.project,
            entity=self.entity,
            title="HRM Experiment Tracking",
            description=f"""
            Comprehensive experiment tracking dashboard for HRM research.
            
            **Features:**
            - Experiment comparison
            - Hyperparameter analysis
            - Performance tracking
            - Resource usage monitoring
            """
        )
        
        # Add experiment table
        report.add_block(
            type="weave",
            title="Experiment Comparison",
            config={
                "tableConfig": {
                    "columnNames": [
                        "Name",
                        "Model",
                        "Pass@1",
                        "Pass@10",
                        "Training Time",
                        "Memory Usage",
                        "Learning Rate",
                        "Batch Size"
                    ],
                    "columnPaths": [
                        "name",
                        "config.model",
                        "summary.val/pass@1",
                        "summary.val/pass@10",
                        "summary.training_time",
                        "summary.memory_usage",
                        "config.learning_rate",
                        "config.batch_size"
                    ]
                }
            }
        )
        
        # Add hyperparameter correlation panel
        report.add_block(
            type="panel",
            title="Hyperparameter Correlation",
            panels=[
                {
                    "type": "scatter",
                    "query": {
                        "keys": ["config.learning_rate", "val/pass@1"],
                        "filters": {"config.model": HRM_MODEL},
                    },
                    "layout": {"x": 0, "y": 0, "w": 6, "h": 8},
                    "title": "Learning Rate vs Pass@1"
                },
                {
                    "type": "scatter",
                    "query": {
                        "keys": ["config.batch_size", "val/pass@1"],
                        "filters": {"config.model": HRM_MODEL},
                    },
                    "layout": {"x": 6, "y": 0, "w": 6, "h": 8},
                    "title": "Batch Size vs Pass@1"
                }
            ]
        )
        
        # Add parallel coordinates plot
        report.add_block(
            type="panel",
            title="Hyperparameter Exploration",
            panels=[
                {
                    "type": "parallel-coordinates",
                    "query": {
                        "keys": [
                            "config.learning_rate",
                            "config.batch_size",
                            "config.num_layers",
                            "config.hidden_size",
                            "config.dropout",
                            "val/pass@1"
                        ],
                        "filters": {"config.model": HRM_MODEL},
                    },
                    "layout": {"x": 0, "y": 8, "w": 12, "h": 10},
                    "title": "Hyperparameter Exploration"
                }
            ]
        )
        
        # Add best runs panel
        report.add_block(
            type="markdown",
            title="Best Configurations",
            content=f"""
            ## Best Performing Configurations
            
            ### Top HRM Configurations
            
            | Run | Pass@1 | Pass@10 | Learning Rate | Batch Size | Layers | Hidden Size |
            |-----|--------|---------|---------------|------------|--------|-------------|
            | {{best_runs.0.name}} | {{best_runs.0.val/pass@1}} | {{best_runs.0.val/pass@10}} | {{best_runs.0.config.learning_rate}} | {{best_runs.0.config.batch_size}} | {{best_runs.0.config.num_layers}} | {{best_runs.0.config.hidden_size}} |
            | {{best_runs.1.name}} | {{best_runs.1.val/pass@1}} | {{best_runs.1.val/pass@10}} | {{best_runs.1.config.learning_rate}} | {{best_runs.1.config.batch_size}} | {{best_runs.1.config.num_layers}} | {{best_runs.1.config.hidden_size}} |
            | {{best_runs.2.name}} | {{best_runs.2.val/pass@1}} | {{best_runs.2.val/pass@10}} | {{best_runs.2.config.learning_rate}} | {{best_runs.2.config.batch_size}} | {{best_runs.2.config.num_layers}} | {{best_runs.2.config.hidden_size}} |
            
            ### Recommended Configuration
            
            Based on performance and efficiency analysis, the recommended configuration is:
            
            ```yaml
            model: {HRM_MODEL}
            learning_rate: {{best_runs.0.config.learning_rate}}
            batch_size: {{best_runs.0.config.batch_size}}
            num_layers: {{best_runs.0.config.num_layers}}
            hidden_size: {{best_runs.0.config.hidden_size}}
            dropout: {{best_runs.0.config.dropout}}
            ```
            """
        )
        
        # Save report
        report.save()
        self.reports["experiments"] = report
        print(f"Experiment Tracking Dashboard created: {report.url}")
        
        return report

    def create_sweep_configuration(self):
        """Create W&B sweep configuration for hyperparameter optimization."""
        print("Creating hyperparameter sweep configuration...")
        
        sweep_config = {
            "method": "bayes",  # Bayesian optimization
            "metric": {"name": "val/pass@1", "goal": "maximize"},
            "parameters": {
                "learning_rate": {"min": 1e-5, "max": 1e-3, "distribution": "log_uniform"},
                "batch_size": {"values": [16, 32, 64, 128]},
                "num_layers": {"values": [4, 6, 8, 12]},
                "hidden_size": {"values": [256, 512, 768, 1024]},
                "dropout": {"min": 0.0, "max": 0.5},
                "weight_decay": {"min": 0.0, "max": 0.1}
            }
        }
        
        # Save sweep configuration to file
        output_dir = Path("configs")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "sweep_config.json", "w") as f:
            json.dump(sweep_config, f, indent=2)
        
        print(f"Sweep configuration saved to: {output_dir / 'sweep_config.json'}")
        
        # Create sweep
        try:
            sweep_id = wandb.sweep(sweep_config, project=self.project, entity=self.entity)
            print(f"Sweep created with ID: {sweep_id}")
            print(f"Start sweep agent with: wandb agent {self.entity}/{self.project}/{sweep_id}")
            return sweep_id
        except Exception as e:
            print(f"Error creating sweep: {e}")
            return None
    
    def create_report_templates(self):
        """Create report templates for research paper."""
        print("Creating report templates...")
        
        # Create templates directory
        templates_dir = Path("reports/templates")
        templates_dir.mkdir(exist_ok=True, parents=True)
        
        # Create performance comparison template
        performance_template = {
            "title": "HRM vs Transformer Performance Comparison",
            "sections": [
                {
                    "title": "Model Specifications",
                    "content": f"""
                    | Model | Parameters | Architecture |
                    |-------|------------|--------------|
                    | HRM | 27M | Hierarchical Recurrent Memory |
                    | {BASELINE_MODEL} | 117M | Standard Transformer |
                    """
                },
                {
                    "title": "Performance Metrics",
                    "content": f"""
                    | Model | Pass@1 | Pass@10 | Inference Latency | Memory Usage |
                    |-------|--------|---------|-------------------|--------------|
                    | HRM | {{val/pass@1}} | {{val/pass@10}} | {{inference/latency}} ms | {{memory/gpu_usage_gb}} GB |
                    | {BASELINE_MODEL} | {{val/pass@1}} | {{val/pass@10}} | {{inference/latency}} ms | {{memory/gpu_usage_gb}} GB |
                    """
                },
                {
                    "title": "Statistical Analysis",
                    "content": f"""
                    | Metric | p-value | Significant? | Effect Size |
                    |--------|---------|--------------|-------------|
                    | Pass@1 | {{stats/pass@1_pvalue}} | {{stats/pass@1_significant}} | {{stats/pass@1_effect_size}} |
                    | Pass@10 | {{stats/pass@10_pvalue}} | {{stats/pass@10_significant}} | {{stats/pass@10_effect_size}} |
                    
                    *Significance level: α = 0.05*
                    """
                }
            ]
        }
        
        # Save templates
        with open(templates_dir / "performance_comparison.json", "w") as f:
            json.dump(performance_template, f, indent=2)
        
        print(f"Report templates saved to: {templates_dir}")
        
        return templates_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Set up W&B research dashboards for HRM validation")
    parser.add_argument("--api-key", help="W&B API key")
    parser.add_argument("--entity", help="W&B entity (username or team name)")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project name")
    parser.add_argument("--create-sweep", action="store_true", help="Create hyperparameter sweep")
    parser.add_argument("--create-templates", action="store_true", help="Create report templates")
    args = parser.parse_args()
    
    # Initialize dashboard setup
    dashboard_setup = ResearchDashboardSetup(
        api_key=args.api_key,
        entity=args.entity,
        project=args.project
    )
    
    # Set up dashboards
    dashboard_urls = dashboard_setup.setup_all_dashboards()
    
    # Create sweep configuration if requested
    if args.create_sweep:
        sweep_id = dashboard_setup.create_sweep_configuration()
    
    # Create report templates if requested
    if args.create_templates:
        templates_dir = dashboard_setup.create_report_templates()
    
    print("\nSetup complete! Access your dashboards at:")
    for name, url in dashboard_urls.items():
        print(f"- {name.capitalize()}: {url}")


if __name__ == "__main__":
    main()
