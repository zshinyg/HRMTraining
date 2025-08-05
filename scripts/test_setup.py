#!/usr/bin/env python
"""
Test script for validating the HRM-CodeGen setup.

This script performs a series of tests to verify that all components
of the Hierarchical Reasoning Model (HRM) for code generation are
properly set up and working together correctly.

Tests include:
- Configuration loading
- Model initialization
- Forward pass with dummy data
- Model saving and loading
- Import verification
- Data pipeline functionality
- Generation capabilities

Run this script before starting training to ensure everything is configured correctly.

Usage:
    python scripts/test_setup.py
    python scripts/test_setup.py --config hrm/configs/mbpp_dev.yaml
    python scripts/test_setup.py --verbose
"""

import argparse
import importlib
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path to import HRM modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import HRM modules (will be tested)
try:
    from hrm.config import HRMConfig, get_default_mbpp_config
    from hrm.model import HRMModel, create_hrm_model
except ImportError as e:
    print(f"ERROR: Failed to import HRM modules: {e}")
    print("Make sure you're running this script from the root directory of the project.")
    sys.exit(1)


class ColoredOutput:
    """Utility class for colored terminal output."""
    
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def header(text):
        return f"{ColoredOutput.HEADER}{text}{ColoredOutput.ENDC}"
    
    @staticmethod
    def info(text):
        return f"{ColoredOutput.BLUE}{text}{ColoredOutput.ENDC}"
    
    @staticmethod
    def success(text):
        return f"{ColoredOutput.GREEN}{text}{ColoredOutput.ENDC}"
    
    @staticmethod
    def warning(text):
        return f"{ColoredOutput.YELLOW}{text}{ColoredOutput.ENDC}"
    
    @staticmethod
    def error(text):
        return f"{ColoredOutput.RED}{text}{ColoredOutput.ENDC}"
    
    @staticmethod
    def bold(text):
        return f"{ColoredOutput.BOLD}{text}{ColoredOutput.ENDC}"


class HRMSetupTester:
    """Test suite for validating HRM setup."""
    
    def __init__(self, config_path: Optional[str] = None, verbose: bool = False):
        """
        Initialize the tester.
        
        Args:
            config_path: Path to the configuration file.
            verbose: Whether to print verbose output.
        """
        self.config_path = config_path
        self.verbose = verbose
        self.config = None
        self.model = None
        self.test_config = None  # Store the test config separately
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = tempfile.mkdtemp(prefix="hrm_test_")
        
        # Test results
        self.results = {}
    
    def __del__(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def run_all_tests(self):
        """Run all tests and return overall success status."""
        print(ColoredOutput.header("\n=== HRM-CodeGen Setup Test ===\n"))
        print(f"Device: {self.device}")
        print(f"Temporary directory: {self.temp_dir}")
        print(f"Configuration: {self.config_path or 'Default'}")
        print(f"Verbose mode: {'Enabled' if self.verbose else 'Disabled'}")
        print("\n" + "=" * 50 + "\n")
        
        # Run tests
        self.test_imports()
        self.test_config_loading()
        self.test_model_initialization()
        self.test_forward_pass()
        self.test_model_save_load()
        self.test_data_pipeline()
        self.test_generation()
        
        # Print summary
        self.print_summary()
        
        # Return overall success
        return all(self.results.values())
    
    def print_summary(self):
        """Print a summary of test results."""
        print(ColoredOutput.header("\n=== Test Summary ===\n"))
        
        all_passed = True
        for test_name, passed in self.results.items():
            status = ColoredOutput.success("PASSED") if passed else ColoredOutput.error("FAILED")
            print(f"{test_name.ljust(30)} {status}")
            if not passed:
                all_passed = False
        
        print("\n" + "=" * 50 + "\n")
        
        if all_passed:
            print(ColoredOutput.success("All tests passed! Your HRM-CodeGen setup is ready for training."))
            print("\nNext steps:")
            print("1. Run data conversion: python scripts/convert_mbpp.py")
            print("2. Start training: python scripts/train.py --config hrm/configs/mbpp_base.yaml")
        else:
            print(ColoredOutput.error("Some tests failed. Please fix the issues before training."))
            print("\nCommon issues:")
            print("- Missing dependencies: Check requirements.txt and install missing packages")
            print("- Configuration errors: Verify your YAML config files")
            print("- CUDA issues: Check your PyTorch installation and GPU drivers")
            print("- Path issues: Make sure you're running from the project root directory")
    
    def test_imports(self):
        """Test that all required modules can be imported."""
        print(ColoredOutput.bold("\n[1/7] Testing imports..."))
        
        try:
            # Core Python libraries
            import json
            import pickle
            import random
            import time
            
            # Scientific computing
            import numpy as np
            
            # PyTorch
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.optim import AdamW
            from torch.utils.data import DataLoader, Dataset
            
            # Try importing transformers if available
            try:
                from transformers import AutoTokenizer
                self.log("  ✓ transformers package available")
            except ImportError:
                self.log("  ⚠ transformers package not available (optional)")
            
            # Try importing wandb if available
            try:
                import wandb
                self.log("  ✓ wandb package available")
            except ImportError:
                self.log("  ⚠ wandb package not available (optional)")
            
            # HRM modules (already imported at the top)
            from hrm.config import HRMConfig, ModelConfig, DataConfig, TrainingConfig
            from hrm.model import HRMModel, create_hrm_model
            from hrm.layers import HighLevelModule, LowLevelModule, PositionalEncoding
            
            print(ColoredOutput.success("  ✓ All required modules imported successfully"))
            self.results["test_imports"] = True
            
        except ImportError as e:
            print(ColoredOutput.error(f"  ✗ Import error: {e}"))
            print(ColoredOutput.warning(f"    Suggestion: pip install {str(e).split()[-1]}"))
            self.results["test_imports"] = False
    
    def test_config_loading(self):
        """Test loading configuration from YAML file."""
        print(ColoredOutput.bold("\n[2/7] Testing configuration loading..."))
        
        try:
            # Create a temporary config file if none provided
            if self.config_path is None:
                # Check if configs directory exists
                configs_dir = os.path.join("hrm", "configs")
                if os.path.exists(configs_dir):
                    # Look for YAML files
                    yaml_files = [f for f in os.listdir(configs_dir) if f.endswith(".yaml")]
                    if yaml_files:
                        self.config_path = os.path.join(configs_dir, yaml_files[0])
                        print(f"  Using found config file: {self.config_path}")
            
            # If still no config path, create a temporary one
            if self.config_path is None:
                self.log("  No config file provided, creating a temporary one")
                temp_config = get_default_mbpp_config()
                temp_config_path = os.path.join(self.temp_dir, "temp_config.yaml")
                temp_config.save(temp_config_path)
                self.config_path = temp_config_path
                self.log(f"  Created temporary config at: {self.config_path}")
            
            # Load the config
            if os.path.exists(self.config_path):
                self.config = HRMConfig.from_yaml(self.config_path)
                print(ColoredOutput.success(f"  ✓ Configuration loaded from {self.config_path}"))
                self.log(f"  Model parameters: {self.config.model.total_params:,}")
                self.log(f"  Hidden dimension: {self.config.model.hidden_dim}")
                self.log(f"  Vocabulary size: {self.config.model.vocab_size}")
                self.results["test_config_loading"] = True
            else:
                print(ColoredOutput.error(f"  ✗ Config file not found: {self.config_path}"))
                print(ColoredOutput.warning(f"    Suggestion: Create the configs directory and add YAML files"))
                
                # Fall back to default config
                self.log("  Falling back to default config")
                self.config = get_default_mbpp_config()
                self.results["test_config_loading"] = False
        
        except Exception as e:
            print(ColoredOutput.error(f"  ✗ Error loading configuration: {e}"))
            traceback.print_exc()
            print(ColoredOutput.warning("    Suggestion: Check YAML syntax and required fields"))
            
            # Fall back to default config
            self.log("  Falling back to default config")
            self.config = get_default_mbpp_config()
            self.results["test_config_loading"] = False
    
    def test_model_initialization(self):
        """Test model initialization."""
        print(ColoredOutput.bold("\n[3/7] Testing model initialization..."))
        
        try:
            # Make sure we have a config
            if self.config is None:
                self.config = get_default_mbpp_config()
            
            # Create a smaller model for testing
            self.test_config = HRMConfig.from_dict(asdict(self.config))
            self.test_config.model.hidden_dim = 128
            self.test_config.model.high_level_dim = 64
            self.test_config.model.low_level_dim = 128
            self.test_config.model.high_level_layers = 1
            self.test_config.model.low_level_layers = 2
            
            # Initialize the model
            start_time = time.time()
            self.model = create_hrm_model(self.test_config)
            init_time = time.time() - start_time
            
            # Move to device
            self.model.to(self.device)
            
            # Print model info
            num_params = sum(p.numel() for p in self.model.parameters())
            print(ColoredOutput.success(f"  ✓ Model initialized with {num_params:,} parameters in {init_time:.2f}s"))
            self.log("  Model structure:")
            if self.verbose:
                for name, param in self.model.named_parameters():
                    self.log(f"    {name}: {param.shape}")
            
            self.results["test_model_initialization"] = True
        
        except Exception as e:
            print(ColoredOutput.error(f"  ✗ Error initializing model: {e}"))
            traceback.print_exc()
            print(ColoredOutput.warning("    Suggestion: Check model architecture and configuration"))
            self.results["test_model_initialization"] = False
    
    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        print(ColoredOutput.bold("\n[4/7] Testing forward pass..."))
        
        try:
            # Make sure we have a model
            if self.model is None:
                print(ColoredOutput.error("  ✗ Model not initialized, skipping forward pass test"))
                self.results["test_forward_pass"] = False
                return
            
            # Create dummy data
            batch_size = 2
            seq_len = 16
            vocab_size = self.model.config.vocab_size
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
            attention_mask = torch.ones_like(input_ids)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Forward pass without labels
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                forward_time = time.time() - start_time
            
            # Check outputs
            logits = outputs["logits"]
            expected_shape = (batch_size, seq_len, vocab_size)
            if logits.shape == expected_shape:
                print(ColoredOutput.success(f"  ✓ Forward pass successful in {forward_time:.4f}s"))
                self.log(f"  Output logits shape: {logits.shape}")
                
                # Forward pass with labels
                with torch.no_grad():
                    outputs_with_labels = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True,
                    )
                
                # Check loss
                if "loss" in outputs_with_labels:
                    loss = outputs_with_labels["loss"]
                    print(ColoredOutput.success(f"  ✓ Loss computation successful: {loss.item():.4f}"))
                    self.results["test_forward_pass"] = True
                else:
                    print(ColoredOutput.error("  ✗ Loss not returned in outputs"))
                    self.results["test_forward_pass"] = False
            else:
                print(ColoredOutput.error(f"  ✗ Incorrect output shape: {logits.shape}, expected {expected_shape}"))
                self.results["test_forward_pass"] = False
        
        except Exception as e:
            print(ColoredOutput.error(f"  ✗ Error during forward pass: {e}"))
            traceback.print_exc()
            print(ColoredOutput.warning("    Suggestion: Check model implementation and input shapes"))
            self.results["test_forward_pass"] = False
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        print(ColoredOutput.bold("\n[5/7] Testing model saving and loading..."))
        
        try:
            # Make sure we have a model
            if self.model is None:
                print(ColoredOutput.error("  ✗ Model not initialized, skipping save/load test"))
                self.results["test_model_save_load"] = False
                return
            
            # Create a save directory
            save_dir = os.path.join(self.temp_dir, "model_save_test")
            os.makedirs(save_dir, exist_ok=True)
            
            # Save the model with its test_config
            save_path = os.path.join(save_dir, "test_model.pt")
            torch.save({
                "model": self.model.state_dict(),
                "config": self.test_config.to_dict() if hasattr(self.test_config, "to_dict") else asdict(self.test_config),
            }, save_path)
            
            print(ColoredOutput.success(f"  ✓ Model saved to {save_path}"))
            
            # Load the model using the saved test_config
            checkpoint = torch.load(save_path, map_location=self.device)
            
            # Create a new model with the same test_config
            loaded_config = HRMConfig.from_dict(checkpoint["config"])
            loaded_model = create_hrm_model(loaded_config)
            loaded_model.to(self.device)
            
            # Load the state dict
            loaded_model.load_state_dict(checkpoint["model"])
            
            print(ColoredOutput.success("  ✓ Model loaded successfully"))
            
            # Compare parameters
            original_params = dict(self.model.named_parameters())
            loaded_params = dict(loaded_model.named_parameters())
            
            if set(original_params.keys()) == set(loaded_params.keys()):
                all_equal = True
                for name in original_params:
                    if not torch.allclose(original_params[name], loaded_params[name]):
                        all_equal = False
                        print(ColoredOutput.error(f"  ✗ Parameter mismatch for {name}"))
                        break
                
                if all_equal:
                    print(ColoredOutput.success("  ✓ Original and loaded parameters match"))
                    self.results["test_model_save_load"] = True
                else:
                    print(ColoredOutput.error("  ✗ Parameter values don't match after loading"))
                    self.results["test_model_save_load"] = False
            else:
                print(ColoredOutput.error("  ✗ Parameter keys don't match after loading"))
                self.results["test_model_save_load"] = False
        
        except Exception as e:
            print(ColoredOutput.error(f"  ✗ Error during model save/load: {e}"))
            traceback.print_exc()
            print(ColoredOutput.warning("    Suggestion: Check model serialization and file permissions"))
            self.results["test_model_save_load"] = False
    
    def test_data_pipeline(self):
        """Test data pipeline with dummy data."""
        print(ColoredOutput.bold("\n[6/7] Testing data pipeline..."))
        
        try:
            # Create dummy dataset
            batch_size = 4
            seq_len = 16
            num_samples = 10
            
            # Create random input IDs and attention masks
            input_ids = torch.randint(0, 1000, (num_samples, seq_len))
            attention_mask = torch.ones_like(input_ids)
            labels = torch.randint(0, 1000, (num_samples, seq_len))
            
            # Create a simple dataset and dataloader
            dataset = TensorDataset(input_ids, attention_mask, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Test iteration
            start_time = time.time()
            for batch_idx, (batch_input_ids, batch_attention_mask, batch_labels) in enumerate(dataloader):
                self.log(f"  Batch {batch_idx + 1}/{len(dataloader)}")
                self.log(f"    Input shape: {batch_input_ids.shape}")
                self.log(f"    Attention mask shape: {batch_attention_mask.shape}")
                self.log(f"    Labels shape: {batch_labels.shape}")
            
            pipeline_time = time.time() - start_time
            
            print(ColoredOutput.success(f"  ✓ Data pipeline test successful in {pipeline_time:.4f}s"))
            print(ColoredOutput.success(f"  ✓ Processed {len(dataloader)} batches of size {batch_size}"))
            
            # Test MBPPDataset import
            try:
                # Try to import the dataset class
                from scripts.train import MBPPDataset
                print(ColoredOutput.success("  ✓ MBPPDataset class imported successfully"))
                
                # Check if data directory exists
                data_dir = os.path.join("data", "mbpp")
                if os.path.exists(data_dir) and any(f.endswith(".bin") for f in os.listdir(data_dir)):
                    print(ColoredOutput.success("  ✓ MBPP data files found"))
                else:
                    print(ColoredOutput.warning("  ⚠ MBPP data files not found"))
                    print(ColoredOutput.info("    Run: python scripts/convert_mbpp.py to download and process data"))
                
                self.results["test_data_pipeline"] = True
            except ImportError:
                print(ColoredOutput.warning("  ⚠ Could not import MBPPDataset, but basic data pipeline works"))
                self.results["test_data_pipeline"] = True
        
        except Exception as e:
            print(ColoredOutput.error(f"  ✗ Error in data pipeline: {e}"))
            traceback.print_exc()
            print(ColoredOutput.warning("    Suggestion: Check DataLoader and Dataset implementations"))
            self.results["test_data_pipeline"] = False
    
    def test_generation(self):
        """Test generation functionality."""
        print(ColoredOutput.bold("\n[7/7] Testing generation functionality..."))
        
        try:
            # Make sure we have a model
            if self.model is None:
                print(ColoredOutput.error("  ✗ Model not initialized, skipping generation test"))
                self.results["test_generation"] = False
                return
            
            # Create dummy input
            batch_size = 1
            seq_len = 8
            vocab_size = self.model.config.vocab_size
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
            attention_mask = torch.ones_like(input_ids)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Test generate method
            with torch.no_grad():
                start_time = time.time()
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=seq_len + 8,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                )
                generation_time = time.time() - start_time
            
            # Check outputs
            expected_min_length = seq_len  # Should at least return the input
            if generated_ids.shape[1] >= expected_min_length:
                print(ColoredOutput.success(f"  ✓ Generation successful in {generation_time:.4f}s"))
                self.log(f"  Input shape: {input_ids.shape}")
                self.log(f"  Generated shape: {generated_ids.shape}")
                
                # Test beam search if available
                try:
                    start_time = time.time()
                    beam_outputs = self.model.beam_search(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=seq_len + 8,
                        num_beams=2,
                    )
                    beam_time = time.time() - start_time
                    
                    print(ColoredOutput.success(f"  ✓ Beam search successful in {beam_time:.4f}s"))
                    self.log(f"  Beam search output shape: {beam_outputs.shape}")
                    
                    self.results["test_generation"] = True
                except (AttributeError, NotImplementedError):
                    print(ColoredOutput.warning("  ⚠ Beam search not implemented, but basic generation works"))
                    self.results["test_generation"] = True
            else:
                print(ColoredOutput.error(f"  ✗ Generated sequence too short: {generated_ids.shape[1]}, expected at least {expected_min_length}"))
                self.results["test_generation"] = False
        
        except Exception as e:
            print(ColoredOutput.error(f"  ✗ Error during generation: {e}"))
            traceback.print_exc()
            print(ColoredOutput.warning("    Suggestion: Check generation logic and parameters"))
            self.results["test_generation"] = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the HRM-CodeGen setup"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Run tests
    tester = HRMSetupTester(
        config_path=args.config,
        verbose=args.verbose,
    )
    
    success = tester.run_all_tests()
    
    # Exit with appropriate status code
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
