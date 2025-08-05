"""
Unit tests for HRM forward pass with code generation adaptations.

This module contains comprehensive tests for the HRM model adaptations
from puzzle solving to code generation, including causal attention,
input handling, and integration with tokenization.
"""

import os
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn.functional as F

# Import our code generation adaptation
from hrm_codegen.model import HRMCodeGenerator
from hrm_codegen.config import CodeGenConfig, load_config

# Import tokenization utilities
from tokenization import get_tokenizer, encode, decode, create_training_batch

# Import MBPP dataset
from datasets.mbpp_loader import MBPPDataset, MBPPConfig


@pytest.fixture
def device():
    """Fixture for the compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def code_gen_config():
    """Fixture for code generation configuration."""
    config_path = os.path.join("configs", "codegen_base.yaml")
    if os.path.exists(config_path):
        return load_config(config_path)
    else:
        # Create default config if file doesn't exist
        return CodeGenConfig()


@pytest.fixture
def hrm_model(code_gen_config, device):
    """Fixture for HRM model with code generation configuration."""
    model = HRMCodeGenerator.from_config(code_gen_config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def dummy_batch(device):
    """Fixture providing a dummy batch for testing."""
    batch_size = 2
    seq_len = 16
    vocab_size = 50257  # GPT-2 vocab size
    
    # Create random input IDs
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), 
        dtype=torch.long, device=device
    )
    
    return {"input_ids": input_ids}


@pytest.fixture
def tokenized_prompts(device):
    """Fixture providing tokenized prompts for testing."""
    prompts = [
        "# Write a function to add two numbers\n\ndef add_numbers(a, b):",
        "# Write a function to check if a number is prime\n\ndef is_prime(n):"
    ]
    
    encoded = encode(prompts, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    return encoded


@pytest.fixture
def mbpp_sample_batch(device):
    """Fixture providing a sample MBPP batch."""
    prompts = [
        "# Write a function to add two numbers",
        "# Write a function to check if a number is prime"
    ]
    
    completions = [
        "def add_numbers(a, b):\n    return a + b",
        "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
    ]
    
    batch = create_training_batch(prompts, completions, max_length=64, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    
    return batch


class TestHRMModelCreation:
    """Tests for HRM model creation with code generation configuration."""
    
    def test_model_creation(self, code_gen_config):
        """Test that HRM model can be created with code generation config."""
        model = HRMCodeGenerator.from_config(code_gen_config)
        assert model is not None
        assert isinstance(model, HRMCodeGenerator)
    
    def test_causal_flag_set(self, hrm_model):
        """Test that causal flag is set in attention layers."""
        # Check H-level attention blocks
        for layer in hrm_model.inner.H_level.layers:
            assert layer.self_attn.causal is True
        
        # Check L-level attention blocks
        for layer in hrm_model.inner.L_level.layers:
            assert layer.self_attn.causal is True
    
    def test_puzzle_emb_len_zero(self, hrm_model):
        """Test that puzzle_emb_len is set to 0 for code generation."""
        assert hrm_model.inner.puzzle_emb_len == 0
    
    def test_q_head_disabled(self, code_gen_config):
        """Test that Q-head can be disabled."""
        # Create model with Q-head disabled
        config = code_gen_config
        config.model["enable_q_head"] = False
        model = HRMCodeGenerator.from_config(config)
        
        # Check that Q-head is None
        assert model.inner.q_head is None
        
        # Create model with Q-head enabled
        config.model["enable_q_head"] = True
        model = HRMCodeGenerator.from_config(config)
        
        # Check that Q-head is not None
        assert model.inner.q_head is not None


class TestForwardPass:
    """Tests for HRM forward pass with code generation adaptations."""
    
    def test_forward_pass_completes(self, hrm_model, dummy_batch):
        """Test that forward pass completes without errors."""
        with torch.no_grad():
            carry, outputs = hrm_model.forward(dummy_batch["input_ids"])
        
        assert carry is not None
        assert "logits" in outputs
    
    def test_output_shapes(self, hrm_model, dummy_batch):
        """Test that output shapes are correct [B,S,V]."""
        batch_size, seq_len = dummy_batch["input_ids"].shape
        vocab_size = hrm_model.config.vocab_size
        
        with torch.no_grad():
            _, outputs = hrm_model.forward(dummy_batch["input_ids"])
        
        logits = outputs["logits"]
        assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_causal_attention(self, hrm_model, device):
        """Test that causal attention is working (future tokens don't affect past)."""
        # Create two sequences that differ only in later positions
        seq1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)
        seq2 = torch.tensor([[1, 2, 3, 4, 9, 10, 11, 12]], device=device)
        
        # Forward pass on both sequences
        with torch.no_grad():
            _, outputs1 = hrm_model.forward(seq1)
            _, outputs2 = hrm_model.forward(seq2)
        
        # Compare logits for the first few positions (should be identical with causal masking)
        logits1 = outputs1["logits"][:, :4, :]  # First 4 positions
        logits2 = outputs2["logits"][:, :4, :]  # First 4 positions
        
        # Logits should be identical for the first 4 positions since future tokens shouldn't affect them
        assert torch.allclose(logits1, logits2, rtol=1e-4, atol=1e-4)
    
    def test_input_ids_handling(self, hrm_model, tokenized_prompts):
        """Test that model works with input_ids (not puzzle_identifiers)."""
        # Model should accept input_ids directly
        with torch.no_grad():
            carry, outputs = hrm_model.forward(tokenized_prompts["input_ids"])
        
        assert "logits" in outputs
        assert outputs["logits"].shape[0] == tokenized_prompts["input_ids"].shape[0]
    
    def test_tokenization_integration(self, hrm_model, device):
        """Test model integration with tokenization module."""
        # Encode a prompt
        prompt = "# Write a function to calculate factorial"
        encoded = encode(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        
        # Forward pass
        with torch.no_grad():
            carry, outputs = hrm_model.forward(input_ids)
        
        # Check that output can be decoded
        next_token_logits = outputs["logits"][0, -1, :]
        next_token = torch.argmax(next_token_logits).unsqueeze(0)
        decoded_token = decode(next_token)
        
        assert isinstance(decoded_token, str)
    
    def test_gradient_flow(self, hrm_model, mbpp_sample_batch):
        """Test gradient flow for training."""
        # Enable gradients
        for param in hrm_model.parameters():
            param.requires_grad = True
        
        # Forward pass
        carry, outputs = hrm_model.forward(mbpp_sample_batch["input_ids"])
        logits = outputs["logits"]
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            mbpp_sample_batch["labels"].view(-1),
            ignore_index=-100
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        grad_exists = False
        for name, param in hrm_model.named_parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                grad_exists = True
                break
        
        assert grad_exists, "No gradients were computed"
    
    def test_mbpp_data_format(self, hrm_model, mbpp_sample_batch):
        """Test that model works with MBPP data formats."""
        # Forward pass with MBPP batch
        carry, outputs = hrm_model.forward(mbpp_sample_batch["input_ids"])
        
        # Compute loss with labels
        logits = outputs["logits"]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            mbpp_sample_batch["labels"].view(-1),
            ignore_index=-100
        )
        
        # Loss should be a valid scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestGenerationCapabilities:
    """Tests for code generation capabilities."""
    
    def test_generate_method(self, hrm_model, device):
        """Test that generate method works."""
        prompt = "# Write a function to add two numbers"
        
        # Generate text
        with torch.no_grad():
            generated_text = hrm_model.generate(
                prompt=prompt,
                max_length=32,
                temperature=0.8,
                do_sample=False  # Deterministic for testing
            )
        
        assert isinstance(generated_text, str)
        assert len(generated_text) > len(prompt)
    
    def test_sampling_strategies(self, hrm_model, device):
        """Test different sampling strategies."""
        prompt = "# Write a function to"
        
        # Test greedy decoding
        with torch.no_grad():
            greedy_text = hrm_model.generate(
                prompt=prompt,
                max_length=20,
                do_sample=False
            )
        
        # Test sampling with temperature
        with torch.no_grad():
            sampled_text = hrm_model.generate(
                prompt=prompt,
                max_length=20,
                temperature=0.8,
                do_sample=True
            )
        
        assert isinstance(greedy_text, str)
        assert isinstance(sampled_text, str)
    
    def test_batch_generation(self, hrm_model, device):
        """Test batch generation capabilities."""
        prompts = [
            "# Write a function to add two numbers",
            "# Write a function to check if a string is a palindrome"
        ]
        
        # Generate text for multiple prompts
        with torch.no_grad():
            generated_texts = hrm_model.generate(
                prompt=prompts,
                max_length=32,
                temperature=0.8,
                do_sample=False
            )
        
        assert isinstance(generated_texts, list)
        assert len(generated_texts) == len(prompts)
        for text in generated_texts:
            assert isinstance(text, str)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
