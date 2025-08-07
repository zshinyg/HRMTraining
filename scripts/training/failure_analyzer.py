#!/usr/bin/env python
"""
Failure Analyzer for Training Jobs

This module provides a comprehensive failure analysis system for training jobs,
implementing pattern recognition, root cause analysis, failure prediction,
and automated reporting to help understand and prevent training failures.

Features:
- Training failure pattern recognition and root cause analysis
- Statistical analysis of failure frequencies and correlations
- Machine learning-based failure prediction
- Failure clustering and categorization
- Trend analysis and early warning systems
- Correlation analysis between failures and system state
- Automated failure reports with visualizations
- Recommendations for preventing similar failures
- Integration with monitoring systems for proactive analysis
- Export of analysis data for external reporting and dashboards

Usage:
    from scripts.training.failure_analyzer import FailureAnalyzer
    
    # Initialize analyzer
    analyzer = FailureAnalyzer(
        experiment_name="hrm_training",
        output_dir="outputs/hrm_training",
        config_path="configs/training_config.yaml",
    )
    
    # Register a new failure
    analyzer.register_failure(
        failure_type="cuda_out_of_memory",
        step=1000,
        traceback_str=traceback.format_exc(),
        system_state=system_state,
    )
    
    # Analyze failures
    analysis_results = analyzer.analyze()
    
    # Generate report
    analyzer.generate_report(format="html", output_path="failure_report.html")
    
    # Get recommendations
    recommendations = analyzer.get_recommendations()
"""

import atexit
import collections
import datetime
import glob
import json
import logging
import math
import os
import pickle
import re
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import yaml

# Optional imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Categories of training failures."""
    MEMORY = "memory"
    COMPUTATION = "computation"
    IO = "io"
    NETWORK = "network"
    HARDWARE = "hardware"
    ENVIRONMENT = "environment"
    DATA = "data"
    MODEL = "model"
    OPTIMIZER = "optimizer"
    DISTRIBUTED = "distributed"
    UNKNOWN = "unknown"


class FailureSeverity(Enum):
    """Severity levels of failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FailureImpact(Enum):
    """Impact levels of failures."""
    ISOLATED = "isolated"
    RECURRING = "recurring"
    SYSTEMATIC = "systematic"
    CASCADING = "cascading"


class AnalysisMethod(Enum):
    """Methods for failure analysis."""
    PATTERN_MATCHING = "pattern_matching"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    CLUSTERING = "clustering"
    CORRELATION = "correlation"
    TREND = "trend"
    HYBRID = "hybrid"


class ReportFormat(Enum):
    """Formats for failure analysis reports."""
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    WANDB = "wandb"
    CUSTOM = "custom"


@dataclass
class FailureEvent:
    """Information about a training failure event."""
    failure_type: str
    timestamp: float = field(default_factory=time.time)
    step: Optional[int] = None
    epoch: Optional[int] = None
    batch_idx: Optional[int] = None
    node_rank: Optional[int] = None
    device: Optional[str] = None
    traceback_str: Optional[str] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    system_state: Optional[Dict[str, Any]] = None
    training_metrics: Optional[Dict[str, Any]] = None
    recovery_actions: List[str] = field(default_factory=list)
    recovery_success: Optional[bool] = None
    recovery_time: Optional[float] = None
    category: Optional[FailureCategory] = None
    severity: Optional[FailureSeverity] = None
    impact: Optional[FailureImpact] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailurePattern:
    """Pattern of recurring failures."""
    pattern_id: str
    failure_types: List[str]
    occurrences: int = 0
    first_occurrence: Optional[float] = None
    last_occurrence: Optional[float] = None
    avg_time_between: Optional[float] = None
    steps: List[int] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    traceback_patterns: List[str] = field(default_factory=list)
    system_state_patterns: Dict[str, Any] = field(default_factory=dict)
    recovery_success_rate: Optional[float] = None
    category: Optional[FailureCategory] = None
    severity: Optional[FailureSeverity] = None
    impact: Optional[FailureImpact] = None
    confidence: float = 0.0
    related_patterns: List[str] = field(default_factory=list)


@dataclass
class FailureTrend:
    """Trend of failures over time."""
    trend_id: str
    failure_types: List[str]
    start_time: float
    end_time: float
    total_failures: int = 0
    failure_rate: Optional[float] = None
    direction: Optional[str] = None  # "increasing", "decreasing", "stable"
    slope: Optional[float] = None
    r_squared: Optional[float] = None
    periodic: bool = False
    period: Optional[float] = None
    forecast: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class FailureCorrelation:
    """Correlation between failures and system state."""
    correlation_id: str
    failure_type: str
    variable: str
    correlation_coefficient: float
    p_value: float
    sample_size: int
    relationship: Optional[str] = None  # "positive", "negative", "none"
    confidence: float = 0.0
    causation_likelihood: float = 0.0
    scatter_points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class FailurePrediction:
    """Prediction of future failures."""
    prediction_id: str
    failure_type: str
    probability: float
    predicted_time: Optional[float] = None
    predicted_step: Optional[int] = None
    confidence: float = 0.0
    features_importance: Dict[str, float] = field(default_factory=dict)
    model_type: Optional[str] = None
    model_accuracy: Optional[float] = None
    warning_threshold: float = 0.7
    critical_threshold: float = 0.9


@dataclass
class FailureRecommendation:
    """Recommendation to prevent failures."""
    recommendation_id: str
    title: str
    description: str
    failure_types: List[str]
    categories: List[FailureCategory]
    priority: int  # 1 (highest) to 5 (lowest)
    estimated_impact: float  # 0.0 to 1.0
    implementation_difficulty: int  # 1 (easiest) to 5 (hardest)
    code_changes: Optional[str] = None
    config_changes: Dict[str, Any] = field(default_factory=dict)
    environment_changes: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Results of failure analysis."""
    timestamp: float = field(default_factory=time.time)
    total_failures: int = 0
    unique_failure_types: int = 0
    failure_counts: Dict[str, int] = field(default_factory=dict)
    category_counts: Dict[str, int] = field(default_factory=dict)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    impact_counts: Dict[str, int] = field(default_factory=dict)
    patterns: List[FailurePattern] = field(default_factory=list)
    trends: List[FailureTrend] = field(default_factory=list)
    correlations: List[FailureCorrelation] = field(default_factory=list)
    predictions: List[FailurePrediction] = field(default_factory=list)
    recommendations: List[FailureRecommendation] = field(default_factory=list)
    clusters: Dict[str, List[int]] = field(default_factory=dict)
    failure_timeline: List[Tuple[float, str]] = field(default_factory=list)
    recovery_success_rate: Optional[float] = None
    avg_recovery_time: Optional[float] = None
    most_common_failure: Optional[str] = None
    most_severe_failure: Optional[str] = None
    most_impactful_failure: Optional[str] = None
    analysis_methods_used: List[str] = field(default_factory=list)
    analysis_duration: Optional[float] = None


@dataclass
class FailureAnalyzerConfig:
    """Configuration for the failure analyzer."""
    # General settings
    enabled: bool = True
    analysis_interval_hours: float = 6.0
    max_failures_to_analyze: int = 1000
    min_failures_for_analysis: int = 5
    
    # Pattern recognition settings
    pattern_recognition_enabled: bool = True
    min_pattern_occurrences: int = 3
    max_pattern_gap_hours: float = 24.0
    traceback_similarity_threshold: float = 0.7
    system_state_similarity_threshold: float = 0.8
    
    # Clustering settings
    clustering_enabled: bool = True
    clustering_method: str = "dbscan"
    max_clusters: int = 10
    
    # Trend analysis settings
    trend_analysis_enabled: bool = True
    trend_min_data_points: int = 5
    trend_window_hours: float = 24.0
    trend_forecast_hours: float = 48.0
    
    # Correlation analysis settings
    correlation_analysis_enabled: bool = True
    correlation_min_sample_size: int = 10
    correlation_significance_threshold: float = 0.05
    max_correlations_per_failure: int = 5
    
    # Prediction settings
    prediction_enabled: bool = True
    prediction_method: str = "random_forest"
    prediction_horizon_hours: float = 24.0
    prediction_min_training_samples: int = 20
    prediction_features: List[str] = field(default_factory=list)
    prediction_target_types: List[str] = field(default_factory=list)
    
    # Recommendation settings
    recommendation_enabled: bool = True
    max_recommendations: int = 10
    recommendation_min_confidence: float = 0.7
    
    # Reporting settings
    report_formats: List[str] = field(default_factory=lambda: ["html", "json"])
    report_directory: str = "reports"
    auto_report_enabled: bool = True
    auto_report_interval_hours: float = 24.0
    include_visualizations: bool = True
    
    # Integration settings
    wandb_integration_enabled: bool = False
    notification_enabled: bool = False
    notification_threshold_severity: str = "high"
    notification_urls: List[str] = field(default_factory=list)
    
    # Advanced settings
    text_analysis_enabled: bool = True
    feature_importance_enabled: bool = True
    anomaly_detection_enabled: bool = True
    time_series_analysis_enabled: bool = True
    
    def __post_init__(self):
        """Initialize default values if not provided."""
        if not self.prediction_features:
            self.prediction_features = [
                "memory_percent",
                "gpu_memory_percent",
                "cpu_percent",
                "disk_percent",
                "batch_size",
                "learning_rate",
                "gradient_norm",
                "loss",
            ]
        
        if not self.prediction_target_types:
            self.prediction_target_types = [
                "cuda_out_of_memory",
                "cpu_out_of_memory",
                "gradient_explosion",
                "nan_loss",
            ]


class FailureAnalyzer:
    """
    Failure Analyzer for Training Jobs
    
    This class provides comprehensive failure analysis for training jobs,
    implementing pattern recognition, root cause analysis, failure prediction,
    and automated reporting to help understand and prevent training failures.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the failure analyzer.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save outputs
            config_path: Path to configuration file (optional)
            config: Configuration dictionary (optional, overrides config_path)
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config(config_path, config)
        
        # Initialize state
        self.failures = []
        self.patterns = []
        self.trends = []
        self.correlations = []
        self.predictions = []
        self.recommendations = []
        self.analysis_results = None
        
        # Counters and stats
        self.last_analysis_time = 0
        self.last_report_time = 0
        self.total_analyses = 0
        
        # ML models
        self.prediction_models = {}
        self.clustering_model = None
        self.vectorizer = None
        self.feature_scaler = None
        
        # Initialize directories
        self._init_directories()
        
        # Load existing data if available
        self._load_existing_data()
        
        # Register exit handler
        atexit.register(self._save_data)
        
        logger.info(f"Failure analyzer initialized for experiment: {experiment_name}")
    
    def _load_config(
        self,
        config_path: Optional[str],
        config_dict: Optional[Dict[str, Any]],
    ) -> FailureAnalyzerConfig:
        """
        Load configuration from file or dictionary.
        
        Args:
            config_path: Path to configuration file
            config_dict: Configuration dictionary
            
        Returns:
            FailureAnalyzerConfig: Configuration object
        """
        # Start with default config
        config = FailureAnalyzerConfig()
        
        # Load from file if provided
        if config_path:
            try:
                with open(config_path, "r") as f:
                    file_config = yaml.safe_load(f)
                
                # Extract failure analyzer config
                if "failure_analyzer" in file_config:
                    analyzer_config = file_config["failure_analyzer"]
                else:
                    analyzer_config = file_config
                
                # Update config with file values
                self._update_config_from_dict(config, analyzer_config)
                
                logger.info(f"Loaded failure analyzer configuration from {config_path}")
            
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        
        # Override with provided config dict
        if config_dict:
            self._update_config_from_dict(config, config_dict)
            logger.info("Applied custom configuration")
        
        return config
    
    def _update_config_from_dict(self, config: FailureAnalyzerConfig, config_dict: Dict[str, Any]):
        """
        Update configuration from dictionary.
        
        Args:
            config: Configuration object to update
            config_dict: Dictionary with configuration values
        """
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    def _init_directories(self):
        """Initialize directories for failure analyzer outputs."""
        try:
            # Create main directories
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create failure analyzer subdirectory
            self.analysis_dir = self.output_dir / "failure_analysis"
            self.analysis_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (self.analysis_dir / "data").mkdir(exist_ok=True)
            (self.analysis_dir / self.config.report_directory).mkdir(exist_ok=True)
            (self.analysis_dir / "models").mkdir(exist_ok=True)
            (self.analysis_dir / "visualizations").mkdir(exist_ok=True)
            
            logger.debug(f"Initialized failure analyzer directories in {self.analysis_dir}")
        
        except Exception as e:
            logger.error(f"Error initializing directories: {e}")
    
    def _load_existing_data(self):
        """Load existing failure data if available."""
        try:
            data_file = self.analysis_dir / "data" / "failures.pkl"
            if data_file.exists():
                with open(data_file, "rb") as f:
                    data = pickle.load(f)
                
                self.failures = data.get("failures", [])
                self.patterns = data.get("patterns", [])
                self.trends = data.get("trends", [])
                self.correlations = data.get("correlations", [])
                self.predictions = data.get("predictions", [])
                self.recommendations = data.get("recommendations", [])
                self.last_analysis_time = data.get("last_analysis_time", 0)
                self.last_report_time = data.get("last_report_time", 0)
                self.total_analyses = data.get("total_analyses", 0)
                
                logger.info(f"Loaded {len(self.failures)} existing failure records")
                
                # Load ML models if available
                self._load_models()
        
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
    
    def _save_data(self):
        """Save failure data to disk."""
        try:
            data_file = self.analysis_dir / "data" / "failures.pkl"
            
            data = {
                "failures": self.failures,
                "patterns": self.patterns,
                "trends": self.trends,
                "correlations": self.correlations,
                "predictions": self.predictions,
                "recommendations": self.recommendations,
                "last_analysis_time": self.last_analysis_time,
                "last_report_time": self.last_report_time,
                "total_analyses": self.total_analyses,
            }
            
            with open(data_file, "wb") as f:
                pickle.dump(data, f)
            
            logger.debug(f"Saved failure data to {data_file}")
            
            # Save ML models if available
            self._save_models()
        
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def _load_models(self):
        """Load ML models if available."""
        if not SKLEARN_AVAILABLE:
            return
        
        try:
            models_dir = self.analysis_dir / "models"
            
            # Load prediction models
            for model_file in models_dir.glob("prediction_*.pkl"):
                failure_type = model_file.stem.replace("prediction_", "")
                with open(model_file, "rb") as f:
                    self.prediction_models[failure_type] = pickle.load(f)
            
            # Load clustering model
            clustering_file = models_dir / "clustering_model.pkl"
            if clustering_file.exists():
                with open(clustering_file, "rb") as f:
                    self.clustering_model = pickle.load(f)
            
            # Load vectorizer
            vectorizer_file = models_dir / "vectorizer.pkl"
            if vectorizer_file.exists():
                with open(vectorizer_file, "rb") as f:
                    self.vectorizer = pickle.load(f)
            
            # Load feature scaler
            scaler_file = models_dir / "feature_scaler.pkl"
            if scaler_file.exists():
                with open(scaler_file, "rb") as f:
                    self.feature_scaler = pickle.load(f)
            
            logger.debug(f"Loaded ML models: {len(self.prediction_models)} prediction models")
        
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
    
    def _save_models(self):
        """Save ML models to disk."""
        if not SKLEARN_AVAILABLE:
            return
        
        try:
            models_dir = self.analysis_dir / "models"
            
            # Save prediction models
            for failure_type, model in self.prediction_models.items():
                model_file = models_dir / f"prediction_{failure_type}.pkl"
                with open(model_file, "wb") as f:
                    pickle.dump(model, f)
            
            # Save clustering model
            if self.clustering_model is not None:
                clustering_file = models_dir / "clustering_model.pkl"
                with open(clustering_file, "wb") as f:
                    pickle.dump(self.clustering_model, f)
            
            # Save vectorizer
            if self.vectorizer is not None:
                vectorizer_file = models_dir / "vectorizer.pkl"
                with open(vectorizer_file, "wb") as f:
                    pickle.dump(self.vectorizer, f)
            
            # Save feature scaler
            if self.feature_scaler is not None:
                scaler_file = models_dir / "feature_scaler.pkl"
                with open(scaler_file, "wb") as f:
                    pickle.dump(self.feature_scaler, f)
            
            logger.debug(f"Saved ML models: {len(self.prediction_models)} prediction models")
        
        except Exception as e:
            logger.error(f"Error saving ML models: {e}")
    
    def register_failure(
        self,
        failure_type: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
        traceback_str: Optional[str] = None,
        exception: Optional[Exception] = None,
        system_state: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
        recovery_actions: Optional[List[str]] = None,
        recovery_success: Optional[bool] = None,
        recovery_time: Optional[float] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ) -> FailureEvent:
        """
        Register a new failure event.
        
        Args:
            failure_type: Type of failure
            step: Current training step
            epoch: Current training epoch
            batch_idx: Current batch index
            traceback_str: String representation of the traceback
            exception: The exception that occurred
            system_state: System state at the time of failure
            training_metrics: Training metrics at the time of failure
            recovery_actions: Actions taken to recover from the failure
            recovery_success: Whether recovery was successful
            recovery_time: Time taken to recover (seconds)
            custom_data: Additional custom data
            
        Returns:
            FailureEvent: The created failure event
        """
        # Extract exception info if provided
        exception_type = None
        exception_message = None
        if exception is not None:
            exception_type = type(exception).__name__
            exception_message = str(exception)
        
        # Create failure event
        failure = FailureEvent(
            failure_type=failure_type,
            step=step,
            epoch=epoch,
            batch_idx=batch_idx,
            traceback_str=traceback_str,
            exception_type=exception_type,
            exception_message=exception_message,
            system_state=system_state or {},
            training_metrics=training_metrics or {},
            recovery_actions=recovery_actions or [],
            recovery_success=recovery_success,
            recovery_time=recovery_time,
            custom_data=custom_data or {},
        )
        
        # Categorize failure
        failure.category = self._categorize_failure(failure)
        
        # Determine severity
        failure.severity = self._determine_severity(failure)
        
        # Determine impact
        failure.impact = self._determine_impact(failure)
        
        # Add to failures list
        self.failures.append(failure)
        
        # Log failure
        logger.info(
            f"Registered failure: {failure_type} at step={step}, epoch={epoch} "
            f"(category={failure.category.value}, severity={failure.severity.value})"
        )
        
        # Save data periodically
        if len(self.failures) % 10 == 0:
            self._save_data()
        
        # Run analysis if enough time has passed
        current_time = time.time()
        if (
            current_time - self.last_analysis_time > self.config.analysis_interval_hours * 3600 and
            len(self.failures) >= self.config.min_failures_for_analysis
        ):
            self.analyze()
        
        return failure
    
    def update_failure(
        self,
        failure_idx: int,
        recovery_actions: Optional[List[str]] = None,
        recovery_success: Optional[bool] = None,
        recovery_time: Optional[float] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing failure event with recovery information.
        
        Args:
            failure_idx: Index of the failure in the failures list
            recovery_actions: Actions taken to recover from the failure
            recovery_success: Whether recovery was successful
            recovery_time: Time taken to recover (seconds)
            custom_data: Additional custom data to update
            
        Returns:
            bool: Whether the update was successful
        """
        if failure_idx < 0 or failure_idx >= len(self.failures):
            logger.error(f"Invalid failure index: {failure_idx}")
            return False
        
        failure = self.failures[failure_idx]
        
        if recovery_actions is not None:
            failure.recovery_actions = recovery_actions
        
        if recovery_success is not None:
            failure.recovery_success = recovery_success
        
        if recovery_time is not None:
            failure.recovery_time = recovery_time
        
        if custom_data is not None:
            failure.custom_data.update(custom_data)
        
        # Re-determine impact based on recovery information
        failure.impact = self._determine_impact(failure)
        
        logger.debug(f"Updated failure {failure_idx}: {failure.failure_type}")
        return True
    
    def _categorize_failure(self, failure: FailureEvent) -> FailureCategory:
        """
        Categorize a failure based on its type and context.
        
        Args:
            failure: Failure event to categorize
            
        Returns:
            FailureCategory: Category of the failure
        """
        failure_type = failure.failure_type.lower()
        traceback_str = failure.traceback_str or ""
        
        # Memory-related failures
        if any(term in failure_type for term in ["memory", "oom", "allocation"]):
            return FailureCategory.MEMORY
        
        # Computation-related failures
        if any(term in failure_type for term in ["nan", "inf", "overflow", "underflow", "gradient"]):
            return FailureCategory.COMPUTATION
        
        # IO-related failures
        if any(term in failure_type for term in ["io", "disk", "file", "permission", "space"]):
            return FailureCategory.IO
        
        # Network-related failures
        if any(term in failure_type for term in ["network", "connection", "timeout", "socket"]):
            return FailureCategory.NETWORK
        
        # Hardware-related failures
        if any(term in failure_type for term in ["hardware", "gpu", "cuda", "device"]):
            return FailureCategory.HARDWARE
        
        # Environment-related failures
        if any(term in failure_type for term in ["environment", "dependency", "version", "import"]):
            return FailureCategory.ENVIRONMENT
        
        # Data-related failures
        if any(term in failure_type for term in ["data", "dataset", "batch", "sample"]):
            return FailureCategory.DATA
        
        # Model-related failures
        if any(term in failure_type for term in ["model", "parameter", "weight", "forward"]):
            return FailureCategory.MODEL
        
        # Optimizer-related failures
        if any(term in failure_type for term in ["optimizer", "learning", "rate", "backward"]):
            return FailureCategory.OPTIMIZER
        
        # Distributed-related failures
        if any(term in failure_type for term in ["distributed", "parallel", "sync", "rank", "world"]):
            return FailureCategory.DISTRIBUTED
        
        # Check traceback for additional clues
        if traceback_str:
            if "CUDA" in traceback_str or "cuda" in traceback_str:
                return FailureCategory.HARDWARE
            if "out of memory" in traceback_str:
                return FailureCategory.MEMORY
            if "No such file" in traceback_str or "Permission denied" in traceback_str:
                return FailureCategory.IO
            if "ImportError" in traceback_str or "ModuleNotFoundError" in traceback_str:
                return FailureCategory.ENVIRONMENT
        
        # Default to unknown
        return FailureCategory.UNKNOWN
    
    def _determine_severity(self, failure: FailureEvent) -> FailureSeverity:
        """
        Determine the severity of a failure.
        
        Args:
            failure: Failure event to analyze
            
        Returns:
            FailureSeverity: Severity of the failure
        """
        # Critical failures that prevent training from continuing
        if failure.failure_type.lower() in [
            "cuda_out_of_memory",
            "cpu_out_of_memory",
            "disk_out_of_space",
            "node_failure",
            "deadlock",
        ]:
            return FailureSeverity.CRITICAL
        
        # High severity failures that significantly impact training
        if failure.failure_type.lower() in [
            "nan_loss",
            "inf_loss",
            "gradient_explosion",
            "checkpoint_corruption",
            "data_corruption",
        ]:
            return FailureSeverity.HIGH
        
        # Medium severity failures that may impact training quality
        if failure.failure_type.lower() in [
            "gradient_vanishing",
            "network_failure",
            "timeout",
            "permission_error",
        ]:
            return FailureSeverity.MEDIUM
        
        # Low severity failures that have minimal impact
        return FailureSeverity.LOW
    
    def _determine_impact(self, failure: FailureEvent) -> FailureImpact:
        """
        Determine the impact of a failure.
        
        Args:
            failure: Failure event to analyze
            
        Returns:
            FailureImpact: Impact of the failure
        """
        # Check if this is a recurring failure type
        failure_type_counts = collections.Counter(f.failure_type for f in self.failures)
        
        # Cascading failures that trigger other failures
        if failure.recovery_success is False and failure.severity == FailureSeverity.CRITICAL:
            return FailureImpact.CASCADING
        
        # Systematic failures that occur regularly
        if failure_type_counts[failure.failure_type] >= 5:
            return FailureImpact.SYSTEMATIC
        
        # Recurring failures that happen occasionally
        if failure_type_counts[failure.failure_type] >= 2:
            return FailureImpact.RECURRING
        
        # Isolated failures that happen once
        return FailureImpact.ISOLATED
    
    def analyze(self) -> AnalysisResult:
        """
        Analyze failure data to identify patterns, trends, and correlations.
        
        Returns:
            AnalysisResult: Results of the analysis
        """
        if len(self.failures) < self.config.min_failures_for_analysis:
            logger.warning(
                f"Not enough failures to analyze. Have {len(self.failures)}, "
                f"need {self.config.min_failures_for_analysis}"
            )
            return None
        
        logger.info(f"Starting failure analysis for {len(self.failures)} failures")
        analysis_start_time = time.time()
        
        # Initialize result
        result = AnalysisResult()
        result.total_failures = len(self.failures)
        
        # Limit number of failures to analyze if needed
        failures_to_analyze = self.failures
        if len(failures_to_analyze) > self.config.max_failures_to_analyze:
            failures_to_analyze = failures_to_analyze[-self.config.max_failures_to_analyze:]
            logger.info(f"Limiting analysis to the most recent {len(failures_to_analyze)} failures")
        
        # Basic statistics
        self._analyze_basic_statistics(failures_to_analyze, result)
        
        # Pattern recognition
        if self.config.pattern_recognition_enabled:
            self._analyze_patterns(failures_to_analyze, result)
            result.analysis_methods_used.append(AnalysisMethod.PATTERN_MATCHING.value)
        
        # Clustering
        if self.config.clustering_enabled and SKLEARN_AVAILABLE:
            self._analyze_clusters(failures_to_analyze, result)
            result.analysis_methods_used.append(AnalysisMethod.CLUSTERING.value)
        
        # Trend analysis
        if self.config.trend_analysis_enabled:
            self._analyze_trends(failures_to_analyze, result)
            result.analysis_methods_used.append(AnalysisMethod.TREND.value)
        
        # Correlation analysis
        if self.config.correlation_analysis_enabled:
            self._analyze_correlations(failures_to_analyze, result)
            result.analysis_methods_used.append(AnalysisMethod.CORRELATION.value)
        
        # Prediction
        if self.config.prediction_enabled and SKLEARN_AVAILABLE:
            self._analyze_predictions(failures_to_analyze, result)
            result.analysis_methods_used.append(AnalysisMethod.MACHINE_LEARNING.value)
        
        # Generate recommendations
        if self.config.recommendation_enabled:
            self._generate_recommendations(result)
        
        # Calculate analysis duration
        result.analysis_duration = time.time() - analysis_start_time
        
        # Update state
        self.analysis_results = result
        self.last_analysis_time = time.time()
        self.total_analyses += 1
        
        # Save data
        self._save_data()
        
        # Generate report if auto-reporting is enabled
        if (
            self.config.auto_report_enabled and
            time.time() - self.last_report_time > self.config.auto_report_interval_hours * 3600
        ):
            for format_name in self.config.report_formats:
                try:
                    format_enum = ReportFormat(format_name)
                    self.generate_report(format=format_enum)
                except ValueError:
                    logger.warning(f"Invalid report format: {format_name}")
            
            self.last_report_time = time.time()
        
        logger.info(
            f"Failure analysis completed in {result.analysis_duration:.2f} seconds. "
            f"Found {len(result.patterns)} patterns, {len(result.trends)} trends, "
            f"{len(result.correlations)} correlations, {len(result.predictions)} predictions, "
            f"and {len(result.recommendations)} recommendations."
        )
        
        return result
    
    def _analyze_basic_statistics(self, failures: List[FailureEvent], result: AnalysisResult):
        """
        Analyze basic statistics of failures.
        
        Args:
            failures: List of failures to analyze
            result: Analysis result to update
        """
        # Count failure types
        failure_types = [f.failure_type for f in failures]
        result.failure_counts = dict(collections.Counter(failure_types))
        result.unique_failure_types = len(result.failure_counts)
        
        # Count categories
        categories = [f.category.value for f in failures]
        result.category_counts = dict(collections.Counter(categories))
        
        # Count severities
        severities = [f.severity.value for f in failures]
        result.severity_counts = dict(collections.Counter(severities))
        
        # Count impacts
        impacts = [f.impact.value for f in failures]
        result.impact_counts = dict(collections.Counter(impacts))
        
        # Create failure timeline
        result.failure_timeline = [(f.timestamp, f.failure_type) for f in failures]
        result.failure_timeline.sort(key=lambda x: x[0])
        
        # Calculate recovery statistics
        recoverable_failures = [f for f in failures if f.recovery_success is not None]
        if recoverable_failures:
            successful_recoveries = sum(1 for f in recoverable_failures if f.recovery_success)
            result.recovery_success_rate = successful_recoveries / len(recoverable_failures)
            
            recovery_times = [f.recovery_time for f in recoverable_failures if f.recovery_time is not None]
            if recovery_times:
                result.avg_recovery_time = sum(recovery_times) / len(recovery_times)
        
        # Identify most common failure
        if result.failure_counts:
            result.most_common_failure = max(result.failure_counts.items(), key=lambda x: x[1])[0]
        
        # Identify most severe failure
        severe_failures = [f for f in failures if f.severity == FailureSeverity.CRITICAL]
        if severe_failures:
            severe_counts = collections.Counter(f.failure_type for f in severe_failures)
            result.most_severe_failure = max(severe_counts.items(), key=lambda x: x[1])[0]
        
        # Identify most impactful failure
        cascading_failures = [f for f in failures if f.impact == FailureImpact.CASCADING]
        if cascading_failures:
            cascading_counts = collections.Counter(f.failure_type for f in cascading_failures)
            result.most_impactful_failure = max(cascading_counts.items(), key=lambda x: x[1])[0]
    
    def _analyze_patterns(self, failures: List[FailureEvent], result: AnalysisResult):
        """
        Analyze patterns in failures.
        
        Args:
            failures: List of failures to analyze
            result: Analysis result to update
        """
        logger.info("Analyzing failure patterns")
        
        # Group failures by type
        failures_by_type = collections.defaultdict(list)
        for f in failures:
            failures_by_type[f.failure_type].append(f)
        
        # Find patterns within each failure type
        patterns = []
        for failure_type, type_failures in failures_by_type.items():
            if len(type_failures) < self.config.min_pattern_occurrences:
                continue
            
            # Sort by timestamp
            type_failures.sort(key=lambda x: x.timestamp)
            
            # Look for repeated patterns in steps, epochs, and system state
            pattern = FailurePattern(
                pattern_id=f"pattern_{failure_type}_{len(patterns)}",
                failure_types=[failure_type],
                occurrences=len(type_failures),
                first_occurrence=type_failures[0].timestamp,
                last_occurrence=type_failures[-1].timestamp,
                steps=[f.step for f in type_failures if f.step is not None],
                epochs=[f.epoch for f in type_failures if f.epoch is not None],
                category=type_failures[0].category,
                severity=type_failures[0].severity,
                impact=type_failures[0].impact,
            )
            
            # Calculate average time between occurrences
            if len(type_failures) > 1:
                time_diffs = [
                    type_failures[i+1].timestamp - type_failures[i].timestamp
                    for i in range(len(type_failures) - 1)
                ]
                pattern.avg_time_between = sum(time_diffs) / len(time_diffs)
            
            # Analyze traceback patterns
            traceback_strs = [f.traceback_str for f in type_failures if f.traceback_str]
            if traceback_strs:
                common_lines = self._find_common_traceback_lines(traceback_strs)
                if common_lines:
                    pattern.traceback_patterns = common_lines
            
            # Analyze system state patterns
            system_states = [f.system_state for f in type_failures if f.system_state]
            if system_states:
                common_state = self._find_common_system_state(system_states)
                if common_state:
                    pattern.system_state_patterns = common_state
            
            # Calculate recovery success rate
            recoverable_failures = [f for f in type_failures if f.recovery_success is not None]
            if recoverable_failures:
                successful_recoveries = sum(1 for f in recoverable_failures if f.recovery_success)
                pattern.recovery_success_rate = successful_recoveries / len(recoverable_failures)
            
            # Set confidence based on number of occurrences and consistency
            pattern.confidence = min(0.5 + (len(type_failures) / 20), 0.95)
            
            patterns.append(pattern)
        
        # Look for cross-type patterns
        if len(failures_by_type) > 1:
            self._find_cross_type_patterns(failures_by_type, patterns)
        
        # Update result
        result.patterns = patterns
        self.patterns = patterns
    
    def _find_common_traceback_lines(self, traceback_strs: List[str]) -> List[str]:
        """
        Find common lines in tracebacks.
        
        Args:
            traceback_strs: List of traceback strings
            
        Returns:
            List[str]: Common traceback lines
        """
        if not traceback_strs:
            return []
        
        # Split tracebacks into lines
        traceback_lines = [tb.split("\n") for tb in traceback_strs]
        
        # Find lines that appear in at least 70% of tracebacks
        all_lines = [line for lines in traceback_lines for line in lines]
        line_counts = collections.Counter(all_lines)
        
        threshold = len(traceback_strs) * self.config.traceback_similarity_threshold
        common_lines = [line for line, count in line_counts.items() if count >= threshold]
        
        # Sort by frequency
        common_lines.sort(key=lambda x: line_counts[x], reverse=True)
        
        # Limit to most relevant lines
        return common_lines[:10]
    
    def _find_common_system_state(self, system_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find common values in system states.
        
        Args:
            system_states: List of system state dictionaries
            
        Returns:
            Dict[str, Any]: Common system state values
        """
        if not system_states:
            return {}
        
        # Find keys that appear in all states
        common_keys = set(system_states[0].keys())
        for state in system_states[1:]:
            common_keys &= set(state.keys())
        
        # Find values that are consistent or show a pattern
        common_state = {}
        for key in common_keys:
            values = [state[key] for state in system_states]
            
            # For numeric values, check if they're all within a range
            if all(isinstance(v, (int, float)) for v in values):
                min_val = min(values)
                max_val = max(values)
                mean_val = sum(values) / len(values)
                
                # If values are consistent or show a trend
                if max_val - min_val < mean_val * 0.2:  # Within 20% of mean
                    common_state[key] = {
                        "mean": mean_val,
                        "min": min_val,
                        "max": max_val,
                        "consistent": True,
                    }
                else:
                    # Check if values show a trend
                    increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
                    decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
                    
                    if increasing or decreasing:
                        common_state[key] = {
                            "mean": mean_val,
                            "min": min_val,
                            "max": max_val,
                            "trend": "increasing" if increasing else "decreasing",
                        }
            
            # For string values, check if they're all the same
            elif all(isinstance(v, str) for v in values):
                if len(set(values)) == 1:
                    common_state[key] = values[0]
                elif len(set(values)) <= 3:  # A few different values
                    value_counts = collections.Counter(values)
                    common_state[key] = {
                        "values": dict(value_counts),
                        "most_common": value_counts.most_common(1)[0][0],
                    }
            
            # For boolean values, check if they're all the same
            elif all(isinstance(v, bool) for v in values):
                if len(set(values)) == 1:
                    common_state[key] = values[0]
                else:
                    true_count = sum(values)
                    common_state[key] = {
                        "true_ratio": true_count / len(values),
                        "most_common": true_count > len(values) / 2,
                    }
        
        return common_state
    
    def _find_cross_type_patterns(
        self,
        failures_by_type: Dict[str, List[FailureEvent]],
        patterns: List[FailurePattern],
    ):
        """
        Find patterns across different failure types.
        
        Args:
            failures_by_type: Dictionary mapping failure types to lists of failures
            patterns: List of patterns to update
        """
        # Look for failures that occur in sequence
        all_failures = [f for failures in failures_by_type.values() for f in failures]
        all_failures.sort(key=lambda x: x.timestamp)
        
        # Look for failures that occur close together in time
        max_gap = self.config.max_pattern_gap_hours * 3600  # Convert to seconds
        
        sequences = []
        current_sequence = [all_failures[0]]
        
        for i in range(1, len(all_failures)):
            if all_failures[i].timestamp - current_sequence[-1].timestamp <= max_gap:
                current_sequence.append(all_failures[i])
            else:
                if len(current_sequence) >= 2:
                    sequences.append(current_sequence)
                current_sequence = [all_failures[i]]
        
        if len(current_sequence) >= 2:
            sequences.append(current_sequence)
        
        # Analyze sequences to find patterns
        for i, sequence in enumerate(sequences):
            if len(sequence) < 2:
                continue
            
            # Get sequence of failure types
            failure_types = [f.failure_type for f in sequence]
            
            # Check if this sequence appears multiple times
            sequence_str = "->".join(failure_types)
            
            # Look for this sequence in other sequences
            occurrences = 0
            for other_sequence in sequences:
                other_types = [f.failure_type for f in other_sequence]
                other_str = "->".join(other_types)
                
                if sequence_str in other_str:
                    occurrences += 1
            
            if occurrences >= self.config.min_pattern_occurrences:
                # Create a cross-type pattern
                pattern = FailurePattern(
                    pattern_id=f"cross_pattern_{i}",
                    failure_types=failure_types,
                    occurrences=occurrences,
                    first_occurrence=sequence[0].timestamp,
                    last_occurrence=sequence[-1].timestamp,
                    steps=[f.step for f in sequence if f.step is not None],
                    epochs=[f.epoch for f in sequence if f.epoch is not None],
                    category=FailureCategory.UNKNOWN,  # Mixed categories
                    severity=max([f.severity for f in sequence], key=lambda x: x.value),
                    impact=max([f.impact for f in sequence], key=lambda x: x.value),
                )
                
                # Calculate average time between first and last failure in sequence
                pattern.avg_time_between = (sequence[-1].timestamp - sequence[0].timestamp) / (len(sequence) - 1)
                
                # Set confidence based on number of occurrences
                pattern.confidence = min(0.4 + (occurrences / 10), 0.9)
                
                patterns.append(pattern)
                
                # Add related pattern IDs
                for p in patterns:
                    if p.pattern_id != pattern.pattern_id and p.failure_types[0] in pattern.failure_types:
                        p.related_patterns.append(pattern.pattern_id)
                        pattern.related_patterns.append(p.pattern_id)
    
    def _analyze_clusters(self, failures: List[FailureEvent], result: AnalysisResult):
        """
        Cluster failures based on their characteristics.
        
        Args:
            failures: List of failures to analyze
            result: Analysis result to update
        """
        if not SKLEARN_AVAILABLE or len(failures) < 10:
            return
        
        logger.info("Clustering failures")
        
        # Extract features for clustering
        features = []
        valid_failures = []
        
        for failure in failures:
            feature_dict = {}
            
            # Add basic features
            feature_dict["failure_type"] = failure.failure_type
            feature_dict["category"] = failure.category.value
            feature_dict["severity"] = failure.severity.value
            feature_dict["impact"] = failure.impact.value
            
            # Add step and epoch if available
            if failure.step is not None:
                feature_dict["step"] = failure.step
            if failure.epoch is not None:
                feature_dict["epoch"] = failure.epoch
            
            # Add system state features if available
            if failure.system_state:
                for key, value in failure.system_state.items():
                    if isinstance(value, (int, float)):
                        feature_dict[f"system_{key}"] = value
            
            # Add training metrics if available
            if failure.training_metrics:
                for key, value in failure.training_metrics.items():
                    if isinstance(value, (int, float)):
                        feature_dict[f"metric_{key}"] = value
            
            # Skip if not enough features
            if len(feature_dict) < 5:
                continue
            
            features.append(feature_dict)
            valid_failures.append(failure)
        
        if not features:
            logger.warning("No valid features for clustering")
            return
        
        # Convert to DataFrame if pandas is available
        if PANDAS_AVAILABLE:
            import pandas as pd
            df = pd.DataFrame(features)
            
            # One-hot encode categorical features
            categorical_cols = ["failure_type", "category", "severity", "impact"]
            for col in categorical_cols:
                if col in df.columns:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
            
            # Fill missing values
            df = df.fillna(df.mean())
            
            # Convert back to numpy array
            feature_matrix = df.values
        else:
            # Simple feature extraction without pandas
            # Extract numeric features only
            numeric_features = []
            for feature_dict in features:
                numeric_dict = {
                    k: v for k, v in feature_dict.items()
                    if isinstance(v, (int, float))
                }
                numeric_features.append(list(numeric_dict.values()))
            
            feature_matrix = np.array(numeric_features)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        self.feature_scaler = scaler
        
        # Apply clustering
        if self.config.clustering_method == "dbscan":
            # DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=3)
            cluster_labels = clustering.fit_predict(scaled_features)
        else:
            # K-means clustering with automatic k selection
            max_k = min(self.config.max_clusters, len(valid_failures) // 3)
            best_k = 2
            best_score = -np.inf
            
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(scaled_features)
                score = -kmeans.inertia_  # Negative inertia as score
                
                # Simple elbow method
                if k > 2:
                    score_improvement = (prev_score - score) / abs(prev_score)
                    if score_improvement < 0.2:  # Less than 20% improvement
                        break
                
                best_k = k
                best_score = score
                prev_score = score
            
            # Apply K-means with best k
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
            self.clustering_model = kmeans
        
        # Group failures by cluster
        clusters = collections.defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label >= 0:  # Ignore noise points (-1)
                clusters[str(label)].append(i)
        
        # Update result
        result.clusters = dict(clusters)
        
        # Visualize clusters if matplotlib is available
        if MATPLOTLIB_AVAILABLE and self.config.include_visualizations:
            self._visualize_clusters(scaled_features, cluster_labels)
    
    def _visualize_clusters(self, features: np.ndarray, labels: np.ndarray):
        """
        Visualize clusters using t-SNE and save the plot.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
        """
        try:
            # Apply t-SNE for dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42)
            reduced_features = tsne.fit_transform(features)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Plot points
            unique_labels = set(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    # Noise points
                    color = [0, 0, 0, 1]  # Black
                
                mask = labels == label
                plt.scatter(
                    reduced_features[mask, 0],
                    reduced_features[mask, 1],
                    c=[color],
                    label=f"Cluster {label}" if label != -1 else "Noise",
                    alpha=0.7,
                )
            
            plt.title("Failure Clusters")
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            viz_dir = self.analysis_dir / "visualizations"
            plt.savefig(viz_dir / "failure_clusters.png")
            plt.close()
            
            logger.debug("Saved cluster visualization")
        
        except Exception as e:
            logger.error(f"Error visualizing clusters: {e}")
    
    def _analyze_trends(self, failures: List[FailureEvent], result: AnalysisResult):
        """
        Analyze trends in failures over time.
        
        Args:
            failures: List of failures to analyze
            result: Analysis result to update
        """
        logger.info("Analyzing failure trends")
        
        # Group failures by type
        failures_by_type = collections.defaultdict(list)
        for f in failures:
            failures_by_type[f.failure_type].append(f)
        
        # Analyze trends for each failure type
        trends = []
        for failure_type, type_failures in failures_by_type.items():
            if len(type_failures) < self.config.trend_min_data_points:
                continue
            
            # Sort by timestamp
            type_failures.sort(key=lambda x: x.timestamp)
            
            # Calculate time range
            start_time = type_failures[0].timestamp
            end_time = type_failures[-1].timestamp
            time_range = end_time - start_time
            
            if time_range < 3600:  # Less than an hour of data
                continue
            
            # Calculate failure rate
            failure_rate = len(type_failures) / (time_range / 3600)  # Failures per hour
            
            # Create trend
            trend = FailureTrend(
                trend_id=f"trend_{failure_type}_{len(trends)}",
                failure_types=[failure_type],
                start_time=start_time,
                end_time=end_time,
                total_failures=len(type_failures),
                failure_rate=failure_rate,
            )
            
            # Analyze trend direction
            self._analyze_trend_direction(type_failures, trend)
            
            # Check for periodicity
            self._analyze_trend_periodicity(type_failures, trend)
            
            # Generate forecast
            self._generate_trend_forecast(type_failures, trend)
            
            trends.append(trend)
        
        # Analyze overall failure trend
        if len(failures) >= self.config.trend_min_data_points:
            # Sort by timestamp
            sorted_failures = sorted(failures, key=lambda x: x.timestamp)
            
            # Calculate time range
            start_time = sorted_failures[0].timestamp
            end_time = sorted_failures[-1].timestamp
            time_range = end_time - start_time
            
            if time_range >= 3600:  # At least an hour of data
                # Calculate overall failure rate
                failure_rate = len(failures) / (time_range / 3600)  # Failures per hour
                
                # Create overall trend
                overall_trend = FailureTrend(
                    trend_id=f"trend_overall_{len(trends)}",
                    failure_types=list(set(f.failure_type for f in failures)),
                    start_time=start_time,
                    end_time=end_time,
                    total_failures=len(failures),
                    failure_rate=failure_rate,
                )
                
                # Analyze trend direction
                self._analyze_trend_direction(sorted_failures, overall_trend)
                
                # Check for periodicity
                self._analyze_trend_periodicity(sorted_failures, overall_trend)
                
                # Generate forecast
                self._generate_trend_forecast(sorted_failures, overall_trend)
                
                trends.append(overall_trend)
        
        # Update result
        result.trends = trends
        self.trends = trends
        
        # Visualize trends if matplotlib is available
        if MATPLOTLIB_AVAILABLE and self.config.include_visualizations and trends:
            self._visualize_trends(trends)
    
    def _analyze_trend_direction(self, failures: List[FailureEvent], trend: FailureTrend):
        """
        Analyze the direction of a failure trend.
        
        Args:
            failures: List of failures to analyze
            trend: Trend to update
        """
        # Get timestamps
        timestamps = [f.timestamp for f in failures]
        
        # Convert to hours from start
        hours = [(t - trend.start_time) / 3600 for t in timestamps]
        
        # Count failures per time window
        window_size = max(1, trend.total_failures // self.config.trend_min_data_points)
        windows = []
        counts = []
        
        for i in range(0, len(failures), window_size):
            window_failures = failures[i:i+window_size]
            if not window_failures:
                continue
            
            window_start = window_failures[0].timestamp
            window_end = window_failures[-1].timestamp
            
            if window_end > window_start:
                window_hours = (window_end - window_start) / 3600
                window_rate = len(window_failures) / window_hours
                
                windows.append((window_start + window_end) / 2)  # Middle of window
                counts.append(window_rate)
        
        if len(windows) < 3:
            trend.direction = "stable"
            return
        
        # Simple linear regression
        x = np.array(windows)
        y = np.array(counts)
        
        # Normalize x to avoid numerical issues
        x_norm = (x - x.min()) / (x.max() - x.min())
        
        # Calculate slope and intercept
        n = len(x_norm)
        slope = (n * np.sum(x_norm * y) - np.sum(x_norm) * np.sum(y)) / (n * np.sum(x_norm**2) - np.sum(x_norm)**2)
        intercept = (np.sum(y) - slope * np.sum(x_norm)) / n
        
        # Calculate R-squared
        y_pred = slope * x_norm + intercept
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        # Update trend
        trend.slope = slope
        trend.r_squared = r_squared
        
        # Determine direction
        if abs(slope) < 0.1 or r_squared < 0.3:
            trend.direction = "stable"
        elif slope > 0:
            trend.direction = "increasing"
        else:
            trend.direction = "decreasing"
        
        # Set confidence based on R-squared
        trend.confidence = min(0.5 + r_squared * 0.5, 0.95)
    
    def _analyze_trend_periodicity(self, failures: List[FailureEvent], trend: FailureTrend):
        """
        Analyze periodicity in a failure trend.
        
        Args:
            failures: List of failures to analyze
            trend: Trend to update
        """
        if len(failures) < 10:
            return
        
        # Get timestamps
        timestamps = [f.timestamp for f in failures]
        
        # Calculate intervals between failures
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not intervals:
            return
        
        # Check for consistent intervals
        mean_interval = sum(intervals) / len(intervals)
        std_interval = (sum((x - mean_interval)**2 for x in intervals) / len(intervals))**0.5
        
        # If standard deviation is low relative to mean, it might be periodic
        if std_interval < mean_interval * 0.5:
            trend.periodic = True
            trend.period = mean_interval
    
    def _generate_trend_forecast(self, failures: List[FailureEvent], trend: FailureTrend):
        """
        Generate a forecast for a failure trend.
        
        Args:
            failures: List of failures to analyze
            trend: Trend to update
        """
        # Simple forecasting based on failure rate and direction
        forecast_hours = self.config.trend_forecast_hours
        current_time = time.time()
        
        # Base forecast on recent failure rate
        recent_cutoff = current_time - (24 * 3600)  # Last 24 hours
        recent_failures = [f for f in failures if f.timestamp >= recent_cutoff]
        
        if recent_failures:
            recent_start = recent_failures[0].timestamp
            recent_end = recent_failures[-1].timestamp
            recent_duration = (recent_end - recent_start) / 3600  # hours
            
            if recent_duration > 0:
                recent_rate = len(recent_failures) / recent_duration
            else:
                recent_rate = trend.failure_rate
        else:
            recent_rate = trend.failure_rate
        
        # Adjust rate based on trend direction
        if trend.direction == "increasing" and trend.slope is not None:
            rate_multiplier = 1 + (trend.slope * 0.1)  # Adjust based on slope
        elif trend.direction == "decreasing" and trend.slope is not None:
            rate_multiplier = 1 + (trend.slope * 0.1)  # Slope is negative
        else:
            rate_multiplier = 1.0
        
        # Generate forecast points
        forecast = []
        for hour in range(1, int(forecast_hours) + 1):
            forecast_time = current_time + (hour * 3600)
            forecast_rate = recent_rate * (rate_multiplier ** hour)
            
            # Ensure rate doesn't go negative
            forecast_rate = max(0, forecast_rate)
            
            forecast.append((forecast_time, forecast_rate))
        
        trend.forecast = forecast
    
    def _visualize_trends(self, trends: List[FailureTrend]):
        """
        Visualize failure trends and save the plot.
        
        Args:
            trends: List of trends to visualize
        """
        try:
            plt.figure(figsize=(12, 8))
            
            for i, trend in enumerate(trends):
                # Plot historical data
                if trend.failure_types[0] == "overall":
                    label = "All Failures"
                    color = "black"
                    linestyle = "-"
                else:
                    label = f"{trend.failure_types[0]}"
                    color = plt.cm.tab10(i % 10)
                    linestyle = "--" if i >= 10 else "-"
                
                # Convert timestamps to datetime for better x-axis
                start_dt = datetime.datetime.fromtimestamp(trend.start_time)
                end_dt = datetime.datetime.fromtimestamp(trend.end_time)
                
                # Plot historical trend line
                plt.plot(
                    [start_dt, end_dt],
                    [trend.failure_rate, trend.failure_rate * (1 + (trend.slope or 0))],
                    color=color,
                    linestyle=linestyle,
                    label=f"{label} ({trend.direction})",
                )
                
                # Plot forecast if available
                if trend.forecast:
                    forecast_times = [datetime.datetime.fromtimestamp(t) for t, _ in trend.forecast]
                    forecast_rates = [r for _, r in trend.forecast]
                    
                    plt.plot(
                        forecast_times,
                        forecast_rates,
                        color=color,
                        linestyle=":",
                        alpha=0.7,
                    )
            
            plt.title("Failure Trends and Forecast")
            plt.xlabel("Time")
            plt.ylabel("Failures per Hour")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis as dates
            plt.gcf().autofmt_xdate()
            
            # Save plot
            viz_dir = self.analysis_dir / "visualizations"
            plt.savefig(viz_dir / "failure_trends.png")
            plt.close()
            
            logger.debug("Saved trend visualization")
        
        except Exception as e:
            logger.error(f"Error visualizing trends: {e}")
    
    def _analyze_correlations(self, failures: List[FailureEvent], result: AnalysisResult):
        """
        Analyze correlations between failures and system state.
        
        Args:
            failures: List of failures to analyze
            result: Analysis result to update
        """
        logger.info("Analyzing failure correlations")
        
        # Group failures by type
        failures_by_type = collections.defaultdict(list)
        for f in failures:
            if f.system_state or f.training_metrics:
                failures_by_type[f.failure_type].append(f)
        
        # Analyze correlations for each failure type
        correlations = []
        for failure_type, type_failures in failures_by_type.items():
            if len(type_failures) < self.config.correlation_min_sample_size:
                continue
            
            # Collect variables from system state and training metrics
            variables = collections.defaultdict(list)
            
            for failure in type_failures:
                # Add system state variables
                if failure.system_state:
                    for key, value in failure.system_state.items():
                        if isinstance(value, (int, float)):
                            variables[f"system_{key}"].append(value)
                
                # Add training metrics
                if failure.training_metrics:
                    for key, value in failure.training_metrics.items():
                        if isinstance(value, (int, float)):
                            variables[f"metric_{key}"].append(value)
            
            # Analyze correlations for variables with enough data
            for var_name, values in variables.items():
                if len(values) < self.config.correlation_min_sample_size:
                    continue
                
                # Calculate correlation with failure frequency
                correlation = self._calculate_correlation(type_failures, var_name, values)
                
                if correlation:
                    correlations.append(correlation)
            
            # Limit correlations per failure type
            if correlations:
                correlations.sort(key=lambda x: abs(x.correlation_coefficient), reverse=True)
                correlations = correlations[:self.config.max_correlations_per_failure]
        
        # Update result
        result.correlations = correlations
        self.correlations = correlations
        
        # Visualize correlations if matplotlib is available
        if MATPLOTLIB_AVAILABLE and self.config.include_visualizations and correlations:
            self._visualize_correlations(correlations)
    
    def _calculate_correlation(
        self,
        failures: List[FailureEvent],
        variable_name: str,
        values: List[float],
    ) -> Optional[FailureCorrelation]:
        """
        Calculate correlation between a variable and failure frequency.
        
        Args:
            failures: List of failures to analyze
            variable_name: Name of the variable
            values: Values of the variable
            
        Returns:
            Optional[FailureCorrelation]: Correlation result or None
        """
        if len(values) < 3:
            return None
        
        try:
            # Calculate correlation coefficient (Pearson's r)
            x = np.array(values)
            
            # Create y values (1 for each failure)
            y = np.ones(len(values))
            
            # Calculate correlation
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x**2)
            sum_y2 = np.sum(y**2)
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
            
            if denominator == 0:
                return None
            
            correlation_coefficient = numerator / denominator
            
            # Calculate p-value (simplified)
            t = correlation_coefficient * np.sqrt((n - 2) / (1 - correlation_coefficient**2))
            p_value = 2 * (1 - self._t_distribution_cdf(abs(t), n - 2))
            
            # Create correlation object
            correlation = FailureCorrelation(
                correlation_id=f"corr_{failures[0].failure_type}_{variable_name}",
                failure_type=failures[0].failure_type,
                variable=variable_name,
                correlation_coefficient=correlation_coefficient,
                p_value=p_value,
                sample_size=n,
            )
            
            # Determine relationship
            if abs(correlation_coefficient) < 0.2:
                correlation.relationship = "none"
            elif correlation_coefficient > 0:
                correlation.relationship = "positive"
            else:
                correlation.relationship = "negative"
            
            # Set confidence based on p-value and coefficient
            if p_value <= self.config.correlation_significance_threshold:
                correlation.confidence = min(0.5 + (abs(correlation_coefficient) * 0.5), 0.95)
                correlation.causation_likelihood = min(0.3 + (abs(correlation_coefficient) * 0.4), 0.8)
            else:
                correlation.confidence = max(0.1, abs(correlation_coefficient) * 0.3)
                correlation.causation_likelihood = max(0.1, abs(correlation_coefficient) * 0.2)
            
            # Store scatter points for visualization
            correlation.scatter_points = list(zip(x, y))
            
            return correlation
        
        except Exception as e:
            logger.error(f"Error calculating correlation for {variable_name}: {e}")
            return None
    
    def _t_distribution_cdf(self, t: float, df: int) -> float:
        """
        Calculate CDF of t-distribution (simplified approximation).
        
        Args:
            t: T-statistic
            df: Degrees of freedom
            
        Returns:
            float: Probability
        """
        # Simple approximation of t-distribution CDF
        x = df / (df + t**2)
        if df >= 3:
            return 1 - 0.5 * x**(df/2)
        return 0.5
    
    def _visualize_correlations(self, correlations: List[FailureCorrelation]):
        """
        Visualize failure correlations and save the plot.
        
        Args:
            correlations: List of correlations to visualize
        """
        try:
            # Create bar chart of correlation coefficients
            plt.figure(figsize=(12, 8))
            
            # Sort by absolute correlation
            sorted_correlations = sorted(
                correlations,
                key=lambda x: abs(x.correlation_coefficient),
                reverse=True,
            )[:10]  # Top 10
            
            variable_names = [self._format_variable_name(c.variable) for c in sorted_correlations]
            coefficients = [c.correlation_coefficient for c in sorted_correlations]
            
            # Create bar colors based on significance
            colors = [
                "darkred" if c.correlation_coefficient < 0 and c.p_value <= 0.05 else
                "darkgreen" if c.correlation_coefficient > 0 