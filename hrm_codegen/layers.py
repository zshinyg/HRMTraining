"""
Modified HRM layers with causal attention support for code generation.

This module extends the original HRM layers from Sapient to support
causal attention masking for autoregressive code generation tasks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------- #
# Ensure the `external/` directory (which contains the vendored Sapient HRM
# code under the package name `sapient_hrm`) is on the Python path. This
# mirrors the logic used in `hrm_codegen/model.py` so that the package can be
# imported without requiring an editable install.
# --------------------------------------------------------------------------- #
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_CUR_DIR))
_EXTERNAL_DIR = os.path.join(_ROOT_DIR, "external")
if _EXTERNAL_DIR not in sys.path:
    sys.path.append(_EXTERNAL_DIR)
    sys.path.append(_ROOT_DIR)

# Import original HRM components (now resolvable as `sapient_hrm`)
from sapient_hrm.models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1Block,
    HierarchicalReasoningModel_ACTV1ReasoningModule,
    HierarchicalReasoningModel_ACTV1Config,
)
from sapient_hrm.models.layers import Attention, CosSin


class HRMCausalBlock(HierarchicalReasoningModel_ACTV1Block):
    """
    Modified HRM block with configurable causal attention.
    
    This class extends the original HRM block to support a configurable
    causal flag that is passed to the attention layer.
    """
    
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config, causal: bool = True) -> None:
        """
        Initialize the HRM causal block.
        
        Args:
            config: HRM configuration
            causal: Whether to use causal attention masking
        """
        # Skip parent's __init__ and call grandparent's __init__
        nn.Module.__init__(self)
        
        # Create attention with configurable causal flag
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=causal
        )
        
        # Create MLP (same as original)
        self.mlp = self._create_mlp(config)
        
        # Store norm epsilon
        self.norm_eps = config.rms_norm_eps
    
    def _create_mlp(self, config: HierarchicalReasoningModel_ACTV1Config):
        """Create MLP layer (extracted for subclassing)."""
        from sapient_hrm.models.layers import SwiGLU
        return SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
    
    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the block.
        
        Args:
            cos_sin: Rotary position encoding
            hidden_states: Input hidden states
            
        Returns:
            Updated hidden states
        """
        # Use parent's forward implementation (unchanged)
        return super().forward(cos_sin, hidden_states)


class HRMCausalReasoningModule(HierarchicalReasoningModel_ACTV1ReasoningModule):
    """
    Modified HRM reasoning module with causal blocks.
    
    This class extends the original HRM reasoning module to use
    causal blocks for autoregressive generation.
    """
    
    @classmethod
    def from_config(cls, config: HierarchicalReasoningModel_ACTV1Config, num_layers: int, causal: bool = True):
        """
        Create a reasoning module from configuration.
        
        Args:
            config: HRM configuration
            num_layers: Number of layers in the module
            causal: Whether to use causal attention masking
            
        Returns:
            HRMCausalReasoningModule instance
        """
        layers = [HRMCausalBlock(config, causal=causal) for _ in range(num_layers)]
        return cls(layers)


def make_hrm_blocks_causal(model: nn.Module, causal: bool = True) -> nn.Module:
    """
    Patch an existing HRM model to use causal attention.
    
    This function modifies the causal flag in all attention layers
    of an existing HRM model to enable or disable causal masking.
    
    Args:
        model: HRM model to patch
        causal: Whether to enable causal attention masking
        
    Returns:
        Patched model
    """
    # Recursively find and modify all Attention modules
    for module in model.modules():
        if isinstance(module, Attention):
            module.causal = causal
    
    return model


def create_causal_hrm_blocks(config: HierarchicalReasoningModel_ACTV1Config, causal: bool = True) -> Tuple[nn.Module, nn.Module]:
    """
    Create causal HRM blocks for H and L levels.
    
    Args:
        config: HRM configuration
        causal: Whether to use causal attention masking
        
    Returns:
        Tuple of (H_level, L_level) modules
    """
    H_level = HRMCausalReasoningModule.from_config(
        config=config,
        num_layers=config.H_layers,
        causal=causal
    )
    
    L_level = HRMCausalReasoningModule.from_config(
        config=config,
        num_layers=config.L_layers,
        causal=causal
    )
    
    return H_level, L_level


def patch_hrm_model_for_code_generation(model: nn.Module) -> nn.Module:
    """
    Patch an HRM model for code generation tasks.
    
    This function applies several modifications to make an HRM model
    suitable for code generation:
    1. Makes all attention blocks causal
    2. Sets puzzle_emb_len to 0
    3. Disables ACT (adaptive computation time)
    
    Args:
        model: HRM model to patch
        
    Returns:
        Patched model
    """
    # Make attention blocks causal
    model = make_hrm_blocks_causal(model, causal=True)
    
    # Set puzzle_emb_len to 0 if it exists
    if hasattr(model, "inner") and hasattr(model.inner, "puzzle_emb_len"):
        model.inner.puzzle_emb_len = 0
    
    # Disable ACT if config exists
    if hasattr(model, "config"):
        if hasattr(model.config, "halt_max_steps"):
            model.config.halt_max_steps = 1
        if hasattr(model.config, "halt_exploration_prob"):
            model.config.halt_exploration_prob = 0.0
    
    return model


class CausalAttentionWrapper(nn.Module):
    """
    Wrapper to make any attention module causal.
    
    This class wraps an existing attention module and forces
    it to use causal masking, regardless of its internal settings.
    """
    
    def __init__(self, attention_module: nn.Module):
        """
        Initialize the wrapper.
        
        Args:
            attention_module: Attention module to wrap
        """
        super().__init__()
        self.attention = attention_module
        
        # Store original forward method
        self._original_forward = attention_module.forward
        
        # Replace forward method with causal version
        def causal_forward(*args, **kwargs):
            # Force causal=True in kwargs
            kwargs["causal"] = True
            return self._original_forward(*args, **kwargs)
        
        # Monkey patch the forward method
        self.attention.forward = causal_forward
    
    def forward(self, *args, **kwargs):
        """Forward pass that ensures causal attention."""
        # Force causal=True
        kwargs["causal"] = True
        return self.attention(*args, **kwargs)


def apply_causal_attention_wrappers(model: nn.Module) -> nn.Module:
    """
    Apply causal attention wrappers to all attention modules.
    
    This is an alternative approach to make_hrm_blocks_causal that
    wraps attention modules instead of modifying them directly.
    
    Args:
        model: Model to modify
        
    Returns:
        Modified model
    """
    # Find all Attention modules
    for name, module in model.named_children():
        if isinstance(module, Attention):
            # Replace with wrapped version
            setattr(model, name, CausalAttentionWrapper(module))
        else:
            # Recursively apply to children
            apply_causal_attention_wrappers(module)
    
    return model
