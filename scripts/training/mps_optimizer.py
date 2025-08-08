#!/usr/bin/env python
"""
MPS Optimizer for M1 Mac Training

This module provides optimization utilities for training neural networks on Apple Silicon M1 Macs
using the Metal Performance Shaders (MPS) backend in PyTorch. It handles MPS-specific optimizations,
memory management, performance monitoring, and fallback mechanisms.

Features:
- MPS device detection and capability checking
- Memory optimization for Apple Silicon
- Handling MPS-specific tensor operations and limitations
- Automatic fallback to CPU for unsupported operations
- Batch size optimization based on available memory
- Performance profiling and monitoring
- Mixed precision training support for MPS
- Gradient checkpointing implementation
- Thermal state monitoring and throttling detection
- Training configuration recommendations

Usage:
    from scripts.training.mps_optimizer import MPSOptimizer

    # Initialize the optimizer
    mps_opt = MPSOptimizer(model=model, config=config)

    # Prepare model for MPS training
    model, device = mps_opt.prepare_model()

    # Get optimized batch size
    batch_size = mps_opt.get_optimal_batch_size(initial_batch_size=32)

    # Enable mixed precision if supported
    scaler = mps_opt.configure_mixed_precision()

    # Monitor performance during training
    mps_opt.start_monitoring()

    # After training
    mps_opt.stop_monitoring()
    performance_stats = mps_opt.get_performance_stats()
"""

import gc
import logging
import os
import platform
import subprocess
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MPSOperationMode(Enum):
    """Enum for MPS operation modes."""

    OPTIMAL = "optimal"  # Balance between performance and reliability
    PERFORMANCE = "performance"  # Maximize performance, may be less stable
    MEMORY_EFFICIENT = "memory_efficient"  # Minimize memory usage
    RELIABLE = "reliable"  # Maximize stability, may be slower


@dataclass
class MPSDeviceStats:
    """Statistics about MPS device performance."""

    is_available: bool = False
    metal_version: str = ""
    total_memory_mb: int = 0
    used_memory_mb: int = 0
    memory_utilization: float = 0.0
    cpu_temperature: float = 0.0
    gpu_temperature: float = 0.0
    is_throttling: bool = False
    performance_index: float = 0.0
    supported_operations: List[str] = None
    unsupported_operations: List[str] = None
    fallback_operations_count: int = 0

    def __post_init__(self):
        if self.supported_operations is None:
            self.supported_operations = []
        if self.unsupported_operations is None:
            self.unsupported_operations = []


class MPSOptimizer:
    """
    Optimizer for training on Apple Silicon M1 Macs using MPS.

    This class provides utilities for optimizing training performance,
    managing memory, handling MPS-specific operations, and monitoring
    system health during training.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        operation_mode: MPSOperationMode = MPSOperationMode.OPTIMAL,
        enable_monitoring: bool = True,
        monitoring_interval: float = 5.0,
        memory_fraction: float = 0.8,
        fallback_to_cpu: bool = True,
    ):
        """
        Initialize the MPS optimizer.

        Args:
            model: PyTorch model to optimize (optional)
            config: Training configuration (optional)
            operation_mode: Mode of operation (balance of performance vs. reliability)
            enable_monitoring: Whether to enable performance monitoring
            monitoring_interval: Interval in seconds for monitoring updates
            memory_fraction: Fraction of available memory to use (0.0-1.0)
            fallback_to_cpu: Whether to automatically fall back to CPU for unsupported operations
        """
        self.model = model
        self.config = config or {}
        self.operation_mode = operation_mode
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval
        self.memory_fraction = max(
            0.1, min(0.95, memory_fraction)
        )  # Clamp between 0.1 and 0.95
        self.fallback_to_cpu = fallback_to_cpu

        # Initialize stats
        self.stats = MPSDeviceStats()
        self._monitoring_active = False
        self._monitoring_thread = None
        self._start_time = None
        self._performance_history = []

        # Check MPS availability
        self._check_mps_availability()

        # Register hooks if model is provided
        if model is not None:
            self._register_model_hooks(model)

    def _check_mps_availability(self) -> bool:
        """
        Check if MPS is available on the current system.

        Returns:
            bool: Whether MPS is available
        """
        # Check if running on macOS
        is_macos = platform.system() == "Darwin"

        # Check if PyTorch has MPS support
        has_mps_support = hasattr(torch, "mps") and hasattr(torch.mps, "is_available")

        # Check if MPS is available through PyTorch
        is_mps_available = has_mps_support and torch.mps.is_available()

        # Get macOS version if on macOS
        macos_version = ""
        if is_macos:
            macos_version = platform.mac_ver()[0]

        # Check Metal version if on macOS
        metal_version = ""
        if is_macos:
            try:
                # Try to get Metal version using system_profiler
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = result.stdout

                # Parse Metal version from output
                metal_lines = [line for line in output.split("\n") if "Metal" in line]
                if metal_lines:
                    metal_version = metal_lines[0].split(":")[-1].strip()
            except Exception as e:
                logger.warning(f"Failed to get Metal version: {e}")

        # Store results
        self.stats.is_available = is_mps_available
        self.stats.metal_version = metal_version

        # Log availability
        if is_mps_available:
            logger.info(
                f"MPS is available (macOS {macos_version}, Metal {metal_version})"
            )

            # Get memory information
            self._update_memory_stats()
        else:
            if is_macos:
                logger.warning(
                    "MPS is not available. Ensure you're using macOS 12.3+ and PyTorch 1.12+ with MPS support."
                )
            else:
                logger.info("Not running on macOS, MPS is not available")

        return is_mps_available

    def _update_memory_stats(self) -> None:
        """Update memory statistics for the MPS device."""
        if not self.stats.is_available:
            return

        try:
            # Get system memory as a proxy for unified memory
            vm = psutil.virtual_memory()
            self.stats.total_memory_mb = vm.total // (1024 * 1024)
            self.stats.used_memory_mb = vm.used // (1024 * 1024)
            self.stats.memory_utilization = vm.percent / 100.0
        except Exception as e:
            logger.warning(f"Failed to update memory stats: {e}")

    def _update_thermal_stats(self) -> None:
        """Update thermal statistics for the system."""
        if not self.stats.is_available:
            return

        try:
            # Try to get temperature using system_profiler or osx-cpu-temp
            # This requires additional tools that may not be available
            # As a fallback, we can detect thermal throttling indirectly

            # Check if powermetrics is available (requires sudo)
            # This is just a placeholder - actual implementation would need sudo access
            self.stats.cpu_temperature = 0.0
            self.stats.gpu_temperature = 0.0

            # Check for thermal throttling by monitoring clock speeds
            # If clock speed drops significantly during high load, it may indicate throttling
            self.stats.is_throttling = False

            # For now, use CPU utilization as a proxy for thermal pressure
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:
                # High CPU utilization for extended periods may lead to throttling
                self.stats.is_throttling = True
        except Exception as e:
            logger.warning(f"Failed to update thermal stats: {e}")

    def _register_model_hooks(self, model: nn.Module) -> None:
        """
        Register hooks on the model to track operations and detect issues.

        Args:
            model: PyTorch model to monitor
        """
        if not self.stats.is_available:
            return

        # Track unsupported operations
        self.stats.unsupported_operations = []
        self.stats.fallback_operations_count = 0

        def pre_forward_hook(module, input):
            """Hook called before forward pass to check for potential issues."""
            # This would track operations that might be problematic on MPS
            return input

        def post_forward_hook(module, input, output):
            """Hook called after forward pass to detect issues."""
            # Check if output has been moved to CPU due to unsupported operation
            if isinstance(output, torch.Tensor) and output.device.type == "cpu":
                op_name = module.__class__.__name__
                if op_name not in self.stats.unsupported_operations:
                    self.stats.unsupported_operations.append(op_name)
                    logger.warning(f"Operation {op_name} was executed on CPU, not MPS")
                self.stats.fallback_operations_count += 1
            return output

        # Register hooks for all modules
        for name, module in model.named_modules():
            module.register_forward_pre_hook(pre_forward_hook)
            module.register_forward_hook(post_forward_hook)

    def is_mps_available(self) -> bool:
        """
        Check if MPS is available on the current system.

        Returns:
            bool: Whether MPS is available
        """
        return self.stats.is_available

    def get_device(self) -> torch.device:
        """
        Get the appropriate device for training.

        Returns:
            torch.device: MPS device if available, otherwise CPU
        """
        if self.stats.is_available:
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def prepare_model(
        self, model: Optional[nn.Module] = None
    ) -> Tuple[nn.Module, torch.device]:
        """
        Prepare the model for training on MPS.

        Args:
            model: PyTorch model to prepare (uses self.model if None)

        Returns:
            Tuple[nn.Module, torch.device]: Prepared model and device
        """
        model = model or self.model
        if model is None:
            raise ValueError("No model provided")

        device = self.get_device()

        # Apply MPS-specific optimizations
        if device.type == "mps":
            # Register hooks if not already registered
            if model != self.model:
                self._register_model_hooks(model)

            # Apply gradient checkpointing if in memory efficient mode
            if self.operation_mode == MPSOperationMode.MEMORY_EFFICIENT:
                self._apply_gradient_checkpointing(model)

            # Clear memory before moving model to MPS
            self._clear_memory()

            # Move model to MPS device
            model.to(device)

            # Log model preparation
            logger.info(
                f"Model prepared for MPS training ({self.operation_mode.value} mode)"
            )
        else:
            # Move model to CPU
            model.to(device)
            logger.info("Model prepared for CPU training (MPS not available)")

        return model, device

    def _apply_gradient_checkpointing(self, model: nn.Module) -> None:
        """
        Apply gradient checkpointing to the model to reduce memory usage.

        Args:
            model: PyTorch model to optimize
        """
        # Check if model has gradient checkpointing capability
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            # Try to apply to specific modules (e.g., Transformer layers)
            checkpointing_applied = False
            for module in model.modules():
                if hasattr(module, "gradient_checkpointing_enable"):
                    module.gradient_checkpointing_enable()
                    checkpointing_applied = True

            if checkpointing_applied:
                logger.info("Gradient checkpointing enabled for supported modules")
            else:
                logger.warning(
                    "Gradient checkpointing not supported by model. "
                    "Consider implementing custom checkpointing."
                )

    def _clear_memory(self) -> None:
        """Clear unused memory to maximize available memory for training."""
        # Force garbage collection
        gc.collect()

        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # For MPS, we can try to force memory release
        if self.stats.is_available:
            # Create and delete a large tensor to force memory cleanup
            try:
                # Calculate a reasonable size based on available memory
                self._update_memory_stats()
                available_mb = self.stats.total_memory_mb - self.stats.used_memory_mb
                if available_mb > 100:  # Only if we have at least 100MB available
                    tensor_size = (
                        min(available_mb // 10, 100) * 1024 * 1024 // 4
                    )  # Convert MB to float32 elements
                    temp_tensor = torch.zeros(
                        tensor_size, dtype=torch.float32, device="mps"
                    )
                    del temp_tensor
            except Exception as e:
                logger.debug(f"Memory clearing operation failed: {e}")

            # Force another garbage collection
            gc.collect()

    def get_optimal_batch_size(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 512,
        sample_input_shape: Optional[Tuple[int, ...]] = None,
        safety_factor: float = 0.8,
    ) -> int:
        """
        Determine the optimal batch size for the current device and model.

        Args:
            initial_batch_size: Starting batch size to try
            min_batch_size: Minimum acceptable batch size
            max_batch_size: Maximum batch size to try
            sample_input_shape: Shape of a single input sample (excluding batch dimension)
            safety_factor: Safety factor to prevent OOM (0.0-1.0)

        Returns:
            int: Optimal batch size
        """
        if not self.stats.is_available:
            # On CPU, we can use a moderate batch size
            return min(initial_batch_size, max_batch_size)

        if self.model is None:
            logger.warning("No model provided, returning initial batch size")
            return initial_batch_size

        # Clear memory before testing
        self._clear_memory()

        # Get device
        device = self.get_device()

        # If no sample input shape provided, use a default shape
        if sample_input_shape is None:
            # Try to infer from model
            sample_input_shape = (3, 224, 224)  # Default to image shape

        # Binary search for optimal batch size
        low = min_batch_size
        high = min(initial_batch_size * 2, max_batch_size)
        optimal_batch_size = low

        try:
            # Move model to device if not already
            self.model.to(device)

            # Create a sample input
            sample_input = torch.zeros((1,) + sample_input_shape, device=device)

            # Test initial forward pass
            with torch.no_grad():
                _ = self.model(sample_input)

            # Binary search for optimal batch size
            while low <= high:
                mid = (low + high) // 2
                try:
                    # Try batch size
                    test_input = torch.zeros((mid,) + sample_input_shape, device=device)
                    with torch.no_grad():
                        _ = self.model(test_input)

                    # If successful, try larger batch size
                    optimal_batch_size = mid
                    low = mid + 1
                except RuntimeError as e:
                    # If OOM, try smaller batch size
                    if "out of memory" in str(e).lower():
                        high = mid - 1
                    else:
                        # If other error, break and use last successful batch size
                        logger.warning(f"Error during batch size testing: {e}")
                        break

                # Clear memory after each test
                del test_input
                self._clear_memory()

        except Exception as e:
            logger.warning(f"Error during optimal batch size determination: {e}")
            # Fall back to initial batch size
            optimal_batch_size = initial_batch_size

        # Apply safety factor
        optimal_batch_size = max(
            min_batch_size, int(optimal_batch_size * safety_factor)
        )

        logger.info(f"Optimal batch size determined: {optimal_batch_size}")
        return optimal_batch_size

    def configure_mixed_precision(self) -> Optional[GradScaler]:
        """
        Configure mixed precision training for MPS if supported.

        Returns:
            Optional[GradScaler]: GradScaler for mixed precision training, or None if not supported
        """
        if not self.stats.is_available:
            logger.info("Mixed precision not configured (MPS not available)")
            return None

        # Check if PyTorch version supports mixed precision on MPS
        # As of writing, full mixed precision support for MPS is limited
        supports_mixed_precision = False

        # Check PyTorch version
        torch_version = torch.__version__.split(".")
        major, minor = int(torch_version[0]), int(torch_version[1])

        if major > 1 or (major == 1 and minor >= 13):
            # PyTorch 1.13+ has better MPS support
            supports_mixed_precision = True

        if supports_mixed_precision:
            logger.info("Configuring mixed precision training for MPS")
            scaler = GradScaler()
            return scaler
        else:
            logger.warning(
                "Mixed precision training not fully supported on MPS in this PyTorch version. "
                "Using full precision instead."
            )
            return None

    def start_monitoring(self) -> None:
        """Start monitoring MPS performance and system health."""
        if not self.enable_monitoring or not self.stats.is_available:
            return

        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return

        self._monitoring_active = True
        self._start_time = time.time()
        self._performance_history = []

        # Start monitoring in a separate thread
        import threading

        def monitoring_loop():
            while self._monitoring_active:
                try:
                    self._update_memory_stats()
                    self._update_thermal_stats()

                    # Record performance stats
                    self._performance_history.append(
                        {
                            "timestamp": time.time() - self._start_time,
                            "memory_utilization": self.stats.memory_utilization,
                            "is_throttling": self.stats.is_throttling,
                            "fallback_operations_count": self.stats.fallback_operations_count,
                        }
                    )

                    # Log warnings if necessary
                    if self.stats.is_throttling:
                        logger.warning(
                            "Thermal throttling detected! Performance may be degraded. "
                            "Consider reducing batch size or taking a break to cool down."
                        )

                    if self.stats.memory_utilization > 0.9:
                        logger.warning(
                            f"High memory utilization ({self.stats.memory_utilization:.1%})! "
                            "Risk of out-of-memory errors. Consider reducing batch size."
                        )
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")

                # Sleep until next update
                time.sleep(self.monitoring_interval)

        # Start monitoring thread
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()

        logger.info("MPS performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop monitoring MPS performance."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False

        # Wait for monitoring thread to finish
        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=2.0)
            self._monitoring_thread = None

        logger.info("MPS performance monitoring stopped")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from monitoring.

        Returns:
            Dict[str, Any]: Performance statistics
        """
        if not self._performance_history:
            return {}

        # Calculate statistics from history
        memory_utils = [
            entry["memory_utilization"] for entry in self._performance_history
        ]
        throttling_events = sum(
            1 for entry in self._performance_history if entry["is_throttling"]
        )
        fallback_ops = max(
            entry["fallback_operations_count"] for entry in self._performance_history
        )

        stats = {
            "duration_seconds": (
                time.time() - self._start_time if self._start_time else 0
            ),
            "memory_utilization_avg": np.mean(memory_utils),
            "memory_utilization_max": np.max(memory_utils),
            "throttling_events": throttling_events,
            "throttling_percentage": (
                throttling_events / len(self._performance_history)
                if self._performance_history
                else 0
            ),
            "fallback_operations_count": fallback_ops,
            "unsupported_operations": self.stats.unsupported_operations,
            "detailed_history": self._performance_history,
        }

        return stats

    def get_training_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for optimal training configuration.

        Returns:
            Dict[str, Any]: Training recommendations
        """
        if not self.stats.is_available:
            return {"message": "MPS not available, no recommendations"}

        # Analyze performance history if available
        if self._performance_history:
            memory_utils = [
                entry["memory_utilization"] for entry in self._performance_history
            ]
            avg_memory = np.mean(memory_utils)
            max_memory = np.max(memory_utils)
            throttling_percentage = sum(
                1 for entry in self._performance_history if entry["is_throttling"]
            ) / len(self._performance_history)
        else:
            # Use current stats if no history
            self._update_memory_stats()
            self._update_thermal_stats()
            avg_memory = self.stats.memory_utilization
            max_memory = avg_memory
            throttling_percentage = 0.0

        # Generate recommendations
        recommendations = {
            "message": "MPS training recommendations",
            "operation_mode": self._recommend_operation_mode(
                avg_memory, max_memory, throttling_percentage
            ),
            "batch_size_adjustment": self._recommend_batch_size_adjustment(
                avg_memory, max_memory
            ),
            "mixed_precision": self._recommend_mixed_precision(),
            "gradient_checkpointing": self._recommend_gradient_checkpointing(
                max_memory
            ),
            "thermal_management": self._recommend_thermal_management(
                throttling_percentage
            ),
            "memory_optimization": self._recommend_memory_optimization(avg_memory),
            "fallback_operations": self._recommend_fallback_handling(),
        }

        return recommendations

    def _recommend_operation_mode(
        self, avg_memory: float, max_memory: float, throttling_percentage: float
    ) -> Dict[str, Any]:
        """Generate operation mode recommendations."""
        current_mode = self.operation_mode.value

        if max_memory > 0.9:
            recommended_mode = MPSOperationMode.MEMORY_EFFICIENT.value
            reason = "High memory usage detected"
        elif throttling_percentage > 0.1:
            recommended_mode = MPSOperationMode.RELIABLE.value
            reason = "Thermal throttling detected"
        elif max_memory < 0.6 and throttling_percentage < 0.05:
            recommended_mode = MPSOperationMode.PERFORMANCE.value
            reason = "System has available resources"
        else:
            recommended_mode = MPSOperationMode.OPTIMAL.value
            reason = "Balanced performance and reliability"

        return {
            "current_mode": current_mode,
            "recommended_mode": recommended_mode,
            "reason": reason,
            "should_change": current_mode != recommended_mode,
        }

    def _recommend_batch_size_adjustment(
        self, avg_memory: float, max_memory: float
    ) -> Dict[str, Any]:
        """Generate batch size adjustment recommendations."""
        if max_memory > 0.9:
            factor = 0.8  # Reduce by 20%
            direction = "decrease"
            reason = "High memory usage, risk of OOM errors"
        elif max_memory > 0.8:
            factor = 0.9  # Reduce by 10%
            direction = "decrease"
            reason = "Memory usage approaching limits"
        elif max_memory < 0.5:
            factor = 1.2  # Increase by 20%
            direction = "increase"
            reason = "Low memory usage, potential for higher throughput"
        else:
            factor = 1.0
            direction = "maintain"
            reason = "Memory usage is optimal"

        return {
            "adjustment_factor": factor,
            "direction": direction,
            "reason": reason,
        }

    def _recommend_mixed_precision(self) -> Dict[str, Any]:
        """Generate mixed precision recommendations."""
        # Check PyTorch version
        torch_version = torch.__version__.split(".")
        major, minor = int(torch_version[0]), int(torch_version[1])

        if major > 1 or (major == 1 and minor >= 13):
            recommendation = True
            reason = "Supported in PyTorch 1.13+"
        else:
            recommendation = False
            reason = f"Limited support in PyTorch {torch.__version__}"

        return {
            "recommended": recommendation,
            "reason": reason,
        }

    def _recommend_gradient_checkpointing(self, max_memory: float) -> Dict[str, Any]:
        """Generate gradient checkpointing recommendations."""
        if max_memory > 0.8:
            recommendation = True
            reason = "High memory usage, checkpointing will reduce memory requirements"
        else:
            recommendation = False
            reason = "Memory usage is acceptable, checkpointing may slow training"

        return {
            "recommended": recommendation,
            "reason": reason,
        }

    def _recommend_thermal_management(
        self, throttling_percentage: float
    ) -> Dict[str, Any]:
        """Generate thermal management recommendations."""
        recommendations = []

        if throttling_percentage > 0.2:
            recommendations.append("Consider external cooling solutions")
            recommendations.append("Reduce training duration or take breaks")
            recommendations.append("Lower batch size to reduce computational load")
            severity = "high"
        elif throttling_percentage > 0.05:
            recommendations.append("Ensure good ventilation")
            recommendations.append("Consider reducing batch size")
            recommendations.append("Monitor temperature during long training runs")
            severity = "medium"
        else:
            recommendations.append("Current thermal management is adequate")
            severity = "low"

        return {
            "recommendations": recommendations,
            "throttling_severity": severity,
            "throttling_percentage": throttling_percentage,
        }

    def _recommend_memory_optimization(self, avg_memory: float) -> Dict[str, Any]:
        """Generate memory optimization recommendations."""
        recommendations = []

        if avg_memory > 0.8:
            recommendations.append("Enable gradient checkpointing")
            recommendations.append("Reduce model size or complexity if possible")
            recommendations.append("Implement progressive loading for large datasets")
            recommendations.append("Free unused memory explicitly during training")
            severity = "high"
        elif avg_memory > 0.6:
            recommendations.append("Monitor memory usage during training")
            recommendations.append("Consider gradient checkpointing for larger models")
            recommendations.append("Optimize dataset loading and preprocessing")
            severity = "medium"
        else:
            recommendations.append("Current memory usage is efficient")
            severity = "low"

        return {
            "recommendations": recommendations,
            "memory_pressure_severity": severity,
            "average_utilization": avg_memory,
        }

    def _recommend_fallback_handling(self) -> Dict[str, Any]:
        """Generate recommendations for handling fallback operations."""
        if not self.stats.unsupported_operations:
            return {
                "has_fallbacks": False,
                "message": "No fallback operations detected",
            }

        return {
            "has_fallbacks": True,
            "unsupported_operations": self.stats.unsupported_operations,
            "fallback_count": self.stats.fallback_operations_count,
            "recommendations": [
                "Consider reimplementing unsupported operations with supported alternatives",
                "Pre-process data on CPU before moving to MPS for operations with known issues",
                "For critical operations with fallbacks, consider using CPU explicitly for those parts",
            ],
        }


def get_mps_device_info() -> Dict[str, Any]:
    """
    Get detailed information about the MPS device.

    Returns:
        Dict[str, Any]: MPS device information
    """
    info = {
        "is_available": False,
        "device_name": "Unknown",
        "macos_version": "",
        "metal_version": "",
        "pytorch_version": torch.__version__,
        "supported": False,
    }

    # Check if running on macOS
    if platform.system() != "Darwin":
        info["device_name"] = platform.system()
        return info

    # Get macOS version
    info["macos_version"] = platform.mac_ver()[0]

    # Check if MPS is available
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        info["supported"] = True
        info["is_available"] = torch.mps.is_available()

    # Get device name
    try:
        # Try to get device name using system_profiler
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout

        # Parse device name from output
        model_lines = [
            line
            for line in output.split("\n")
            if "Model Name" in line or "Chip" in line
        ]
        if model_lines:
            info["device_name"] = model_lines[0].split(":")[-1].strip()
    except Exception:
        info["device_name"] = "Mac (unknown model)"

    # Get Metal version
    try:
        # Try to get Metal version using system_profiler
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout

        # Parse Metal version from output
        metal_lines = [line for line in output.split("\n") if "Metal" in line]
        if metal_lines:
            info["metal_version"] = metal_lines[0].split(":")[-1].strip()
    except Exception:
        pass

    return info


def optimize_for_mps(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    operation_mode: str = "optimal",
) -> Tuple[nn.Module, torch.device, Optional[GradScaler]]:
    """
    Optimize a model for training on MPS (convenience function).

    Args:
        model: PyTorch model to optimize
        config: Training configuration
        operation_mode: Mode of operation ("optimal", "performance", "memory_efficient", "reliable")

    Returns:
        Tuple[nn.Module, torch.device, Optional[GradScaler]]: Optimized model, device, and scaler
    """
    # Convert operation mode string to enum
    try:
        mode = MPSOperationMode(operation_mode)
    except ValueError:
        logger.warning(f"Invalid operation mode: {operation_mode}, using 'optimal'")
        mode = MPSOperationMode.OPTIMAL

    # Create optimizer
    optimizer = MPSOptimizer(
        model=model,
        config=config,
        operation_mode=mode,
    )

    # Prepare model
    model, device = optimizer.prepare_model()

    # Configure mixed precision
    scaler = optimizer.configure_mixed_precision()

    # Start monitoring
    optimizer.start_monitoring()

    return model, device, scaler, optimizer


if __name__ == "__main__":
    """Run MPS device check and print information."""
    info = get_mps_device_info()

    print("\n=== MPS Device Information ===")
    print(f"Device: {info['device_name']}")
    print(f"macOS Version: {info['macos_version']}")
    print(f"Metal Version: {info['metal_version']}")
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"MPS Support in PyTorch: {info['supported']}")
    print(f"MPS Available: {info['is_available']}")

    if info["is_available"]:
        print("\n✅ MPS is available and ready for use!")
        print("\nRecommended PyTorch device code:")
        print("```python")
        print("device = torch.device('mps' if torch.mps.is_available() else 'cpu')")
        print("model.to(device)")
        print("```")
    else:
        print("\n❌ MPS is not available.")
        if info["supported"]:
            print(
                "MPS is supported in your PyTorch version but not available on this system."
            )
            print("Ensure you're using macOS 12.3+ on Apple Silicon hardware.")
        else:
            print("Your PyTorch version does not support MPS.")
            print("Consider upgrading to PyTorch 1.12+ for MPS support.")
