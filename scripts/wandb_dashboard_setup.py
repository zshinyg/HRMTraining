#!/usr/bin/env python
"""
W&B Dashboard Setup Script for HRM Training Monitoring

This script initializes a Weights & Biases run with custom panels and alerting
for monitoring HRM model training on Apple Silicon M1 hardware.

Usage:
    python scripts/wandb_dashboard_setup.py --project hrm-codegen --run-name m1-27m-training
    python scripts/wandb_dashboard_setup.py --project hrm-codegen --run-name m1-27m-training --entity my-team
    python scripts/wandb_dashboard_setup.py --project hrm-codegen --config configs/m1_optimized_training.yaml
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import monitoring_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.monitoring_utils import (
    init_monitoring,
    create_custom_charts,
    is_wandb_authenticated,
    WANDB_AVAILABLE
)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Set up W&B dashboard for HRM training monitoring"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="hrm-codegen",
        help="W&B project name"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        default="m1-27m-training",
        help="W&B run name"
    )
    
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity/team name"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file with alert thresholds"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dict containing configuration
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

def extract_thresholds(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract monitoring thresholds from config.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dict containing threshold values
    """
    thresholds = {}
    
    # Try to extract from monitoring section
    if "monitoring" in config and "thresholds" in config["monitoring"]:
        return config["monitoring"]["thresholds"]
    
    # Try to extract from safety section
    if "safety" in config:
        if "loss_watchdog" in config["safety"] and "threshold" in config["safety"]["loss_watchdog"]:
            thresholds["loss"] = config["safety"]["loss_watchdog"]["threshold"]
        
        if "detect_anomaly" in config["safety"]:
            thresholds["nan_detected"] = config["safety"]["detect_anomaly"]
    
    # Extract memory limits
    if "memory" in config and "max_memory_mb" in config["memory"]:
        thresholds["memory_usage_mb"] = config["memory"]["max_memory_mb"] * 0.9  # 90% of max
    
    # Extract from system section
    if "system" in config and "memory_limit_mb" in config["system"]:
        thresholds["memory_usage_mb"] = config["system"]["memory_limit_mb"] * 0.9  # 90% of max
    
    return thresholds

def main() -> None:
    """Main function."""
    args = parse_args()
    
    # Check if W&B is available
    if not WANDB_AVAILABLE:
        logger.error("W&B not installed. Please install wandb package: pip install wandb")
        sys.exit(1)
    
    # Check if W&B is authenticated
    if not is_wandb_authenticated():
        logger.error("W&B not authenticated. Please run 'wandb login' or set WANDB_API_KEY environment variable")
        sys.exit(1)
    
    # Load config if provided
    config = None
    thresholds = None
    if args.config:
        config = load_config(args.config)
        thresholds = extract_thresholds(config)
        logger.info(f"Extracted thresholds: {thresholds}")
    
    # Initialize W&B
    logger.info(f"Initializing W&B run: project={args.project}, run={args.run_name}")
    wandb_initialized = init_monitoring(
        project_name=args.project,
        run_name=args.run_name
    )
    
    if not wandb_initialized:
        logger.error("Failed to initialize W&B")
        sys.exit(1)
    
    # Import wandb here to ensure it's only used after checking availability
    import wandb
    
    # Update W&B config with entity if provided
    if args.entity:
        wandb.run.entity = args.entity
    
    # Create custom charts
    create_custom_charts()
    
    # Print run URL
    run_url = wandb.run.get_url()
    
    logger.info("=" * 80)
    logger.info(f"W&B dashboard created successfully!")
    logger.info(f"Dashboard URL: {run_url}")
    logger.info("=" * 80)
    
    # Print instructions for sharing
    logger.info("Share this URL with stakeholders for real-time training monitoring.")
    logger.info("The dashboard will update as soon as training metrics are logged.")
    
    # Finish the run
    wandb.finish()
    logger.info("Dashboard setup complete. Exiting.")

if __name__ == "__main__":
    main()
