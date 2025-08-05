"""
Basic tests for the configuration system without HRM model dependencies.

This module tests the configuration system for HRM code generation adaptation
without requiring the full HRM model or flash-attn dependencies.
"""

import os
import tempfile
from typing import Dict, Any

import pytest
import yaml

# Import config functionality
# NOTE: Use the lightweight standalone config implementation to avoid heavy
# HRM / flash-attn dependencies during unit testing.
from hrm_codegen.config_standalone import (
    CodeGenConfig,
    load_config,
    merge_configs,
    get_default_config,
    create_runtime_config,
    RuntimeConfig,
)


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config_dict = {
        "task": "codegen",
        "tokenizer_name": "gpt2",
        "model": {
            "name": "HierarchicalReasoningModel_ACTV1",
            "causal": True,
            "hidden_size": 512,
            "num_heads": 8,
            "puzzle_emb_ndim": 0,
        },
        "data": {
            "batch_size": 4,
            "seq_len": 256,
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name
    
    yield temp_path
    
    # Clean up
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestConfigCreation:
    """Test creating configurations with different methods."""
    
    def test_default_config(self):
        """Test creating a default configuration."""
        config = CodeGenConfig()
        assert config.task == "codegen"
        assert config.tokenizer_name == "gpt2"
        assert config.model["causal"] is True
        assert config.model["puzzle_emb_ndim"] == 0
    
    def test_get_default_config(self):
        """Test get_default_config utility."""
        config = get_default_config()
        assert isinstance(config, CodeGenConfig)
        assert config.task == "codegen"
    
    def test_custom_config(self):
        """Test creating a custom configuration."""
        config = CodeGenConfig(
            task="codegen",
            model={
                "hidden_size": 1024,
                "num_heads": 16,
            }
        )
        assert config.model["hidden_size"] == 1024
        assert config.model["num_heads"] == 16
        # Default values should still be present
        assert config.model["causal"] is True
    
    def test_load_config(self, temp_config_file):
        """Test loading configuration from a file."""
        config = load_config(temp_config_file)
        assert config.task == "codegen"
        assert config.model["hidden_size"] == 512
        assert config.model["num_heads"] == 8


class TestConfigValidation:
    """Test configuration validation rules."""
    
    def test_validate_task(self):
        """Test task validation."""
        # Valid tasks
        CodeGenConfig(task="codegen")
        CodeGenConfig(task="puzzle")
        
        # Invalid task
        with pytest.raises(ValueError, match="Task .* not supported"):
            CodeGenConfig(task="invalid")
    
    def test_validate_causal_for_codegen(self):
        """Test that causal must be True for code generation."""
        # Valid: causal=True for codegen
        CodeGenConfig(task="codegen", model={"causal": True})
        
        # Invalid: causal=False for codegen
        with pytest.raises(ValueError, match="Causal must be True for code generation"):
            CodeGenConfig(task="codegen", model={"causal": False})
    
    def test_validate_puzzle_emb_ndim_for_codegen(self):
        """Test that puzzle_emb_ndim must be 0 for code generation."""
        # Valid: puzzle_emb_ndim=0 for codegen
        CodeGenConfig(task="codegen", model={"puzzle_emb_ndim": 0})
        
        # Invalid: puzzle_emb_ndim>0 for codegen
        with pytest.raises(ValueError, match="puzzle_emb_ndim must be 0 for code generation"):
            CodeGenConfig(task="codegen", model={"puzzle_emb_ndim": 10})


class TestConfigConversion:
    """Test configuration conversion methods."""
    
    def test_to_hrm_config(self):
        """Test converting to HRM config format."""
        config = CodeGenConfig(
            model={
                "hidden_size": 768,
                "num_heads": 12,
                "causal": True,
                "enable_q_head": False,
            },
            data={
                "batch_size": 8,
                "seq_len": 512,
            }
        )
        
        hrm_config = config.to_hrm_config()
        
        # Check that HRM config has expected keys
        assert "hidden_size" in hrm_config
        assert "num_heads" in hrm_config
        assert "batch_size" in hrm_config
        assert "seq_len" in hrm_config
        
        # Check that code generation specific keys are removed
        assert "causal" not in hrm_config
        assert "enable_q_head" not in hrm_config
        
        # Check values
        assert hrm_config["hidden_size"] == 768
        assert hrm_config["num_heads"] == 12
        assert hrm_config["batch_size"] == 8
        assert hrm_config["seq_len"] == 512
    
    def test_to_hrm_config_with_overrides(self):
        """Test converting to HRM config with overrides."""
        config = CodeGenConfig(
            data={
                "batch_size": 8,
                "seq_len": 512,
            }
        )
        
        # Override batch_size and seq_len
        hrm_config = config.to_hrm_config(batch_size=16, seq_len=1024)
        
        assert hrm_config["batch_size"] == 16
        assert hrm_config["seq_len"] == 1024


class TestConfigMerging:
    """Test configuration merging functionality."""
    
    def test_merge_configs(self):
        """Test merging configurations."""
        base_config = CodeGenConfig(
            model={
                "hidden_size": 768,
                "num_heads": 12,
            },
            data={
                "batch_size": 8,
            }
        )
        
        override_dict = {
            "model": {
                "hidden_size": 1024,
            },
            "training": {
                "max_steps": 5000,
            }
        }
        
        merged_config = merge_configs(base_config, override_dict)
        
        # Check that overrides are applied
        assert merged_config.model["hidden_size"] == 1024
        assert merged_config.training["max_steps"] == 5000
        
        # Check that non-overridden values are preserved
        assert merged_config.model["num_heads"] == 12
        assert merged_config.data["batch_size"] == 8
    
    def test_deep_merge(self):
        """Test deep merging of nested dictionaries."""
        base_config = CodeGenConfig(
            model={
                "hidden_size": 768,
            },
            training={
                "optimizer": {
                    "name": "AdamW",
                    "lr": 5e-5,
                    "weight_decay": 0.01,
                }
            }
        )
        
        override_dict = {
            "training": {
                "optimizer": {
                    "lr": 1e-4,
                }
            }
        }
        
        merged_config = merge_configs(base_config, override_dict)
        
        # Check that deep override is applied
        assert merged_config.training["optimizer"]["lr"] == 1e-4
        
        # Check that non-overridden nested values are preserved
        assert merged_config.training["optimizer"]["name"] == "AdamW"
        assert merged_config.training["optimizer"]["weight_decay"] == 0.01


class TestRuntimeConfig:
    """Test runtime configuration functionality."""
    
    def test_create_runtime_config(self):
        """Test creating a runtime configuration."""
        runtime_config = create_runtime_config()
        
        assert isinstance(runtime_config, RuntimeConfig)
        assert isinstance(runtime_config.config, CodeGenConfig)
        assert runtime_config.device == "cuda"
        assert runtime_config.precision == "bf16-mixed"
        assert runtime_config.seed == 42
    
    def test_create_runtime_config_with_overrides(self):
        """Test creating a runtime configuration with overrides."""
        runtime_config = create_runtime_config(
            device="cpu",
            seed=123,
            config={
                "model": {
                    "hidden_size": 1024,
                }
            }
        )
        
        assert runtime_config.device == "cpu"
        assert runtime_config.seed == 123
        assert runtime_config.config.model["hidden_size"] == 1024
    
    def test_create_runtime_config_from_file(self, temp_config_file):
        """Test creating a runtime configuration from a file."""
        runtime_config = create_runtime_config(config_path=temp_config_file)
        
        assert isinstance(runtime_config.config, CodeGenConfig)
        assert runtime_config.config.model["hidden_size"] == 512
        assert runtime_config.config.model["num_heads"] == 8
        assert runtime_config.config_path == temp_config_file


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
