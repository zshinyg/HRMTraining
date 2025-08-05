#!/usr/bin/env python3
"""
Sapient HRM Architecture Explorer

This script explores the Hierarchical Reasoning Model (HRM) architecture from Sapient Inc.
without requiring training dependencies. It analyzes the model structure, parameter counts,
and configuration details to aid in adaptation for code generation tasks.
"""

import os
import sys
import json
import yaml
import importlib
from typing import Dict, Any, Optional
import inspect

# Add the Sapient HRM directory to Python path
SAPIENT_HRM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "external", "sapient-hrm"))
sys.path.insert(0, SAPIENT_HRM_DIR)

# Try to import torch - required for model inspection
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} available")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Limited analysis will be performed.")

# Utility functions
def count_parameters(model):
    """Count trainable parameters in a PyTorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model_class(identifier: str, prefix: str = "models."):
    """Load a model class from a string identifier (module_path@class_name)"""
    module_path, class_name = identifier.split('@')
    
    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    
    return cls

def load_yaml_config(config_path):
    """Load a YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model_structure(model):
    """Get a dictionary representation of model structure"""
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    result = {}
    for name, module in model.named_children():
        result[name] = {
            "type": module.__class__.__name__,
            "parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)
        }
        if hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
            result[name]["num_layers"] = len(module.layers)
    
    return result

def create_dummy_input(model_config):
    """Create dummy input tensors based on model configuration"""
    if not TORCH_AVAILABLE:
        return None
    
    batch_size = 2
    seq_len = model_config.get("seq_len", 64)
    
    # Create dummy inputs
    inputs = torch.randint(0, model_config.get("vocab_size", 1000), (batch_size, seq_len))
    puzzle_identifiers = torch.zeros(batch_size, dtype=torch.int64)
    
    return {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_identifiers,
        "labels": torch.randint(0, model_config.get("vocab_size", 1000), (batch_size, seq_len))
    }

def main():
    """Main function to explore the Sapient HRM architecture"""
    print(f"Exploring Sapient HRM in: {SAPIENT_HRM_DIR}")
    
    # Step 1: Load configuration
    config_dir = os.path.join(SAPIENT_HRM_DIR, "config")
    main_config_path = os.path.join(config_dir, "cfg_pretrain.yaml")
    
    try:
        main_config = load_yaml_config(main_config_path)
        print("\n=== Main Configuration ===")
        print(f"Default architecture: {main_config.get('defaults', [{'arch': 'unknown'}])[0].get('arch')}")
        
        # Load architecture config
        arch_name = main_config.get('defaults', [{'arch': 'hrm_v1'}])[0].get('arch')
        arch_config_path = os.path.join(config_dir, "arch", f"{arch_name}.yaml")
        arch_config = load_yaml_config(arch_config_path)
        
        print("\n=== Architecture Configuration ===")
        for key, value in arch_config.items():
            print(f"{key}: {value}")
        
        # Combine configs for model instantiation
        combined_config = {**main_config, "arch": arch_config}
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        combined_config = None
    
    # Step 2: Try to instantiate model
    model = None
    if TORCH_AVAILABLE and combined_config:
        try:
            # Extract model class name from config
            model_class_name = arch_config.get("name")
            model_class = load_model_class(model_class_name)
            
            # Create a minimal config for model instantiation
            model_config = {
                "batch_size": 2,
                "seq_len": 64,
                "puzzle_emb_ndim": arch_config.get("puzzle_emb_ndim", 512),
                "num_puzzle_identifiers": 10,
                "vocab_size": 1000,
                "H_cycles": arch_config.get("H_cycles", 2),
                "L_cycles": arch_config.get("L_cycles", 2),
                "H_layers": arch_config.get("H_layers", 4),
                "L_layers": arch_config.get("L_layers", 4),
                "hidden_size": arch_config.get("hidden_size", 512),
                "expansion": arch_config.get("expansion", 4),
                "num_heads": arch_config.get("num_heads", 8),
                "pos_encodings": arch_config.get("pos_encodings", "rope"),
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "halt_max_steps": arch_config.get("halt_max_steps", 16),
                "halt_exploration_prob": arch_config.get("halt_exploration_prob", 0.1),
                "forward_dtype": "float32"  # Use float32 for CPU compatibility
            }
            
            # Instantiate model
            print("\n=== Instantiating Model ===")
            model = model_class(model_config)
            print(f"Model successfully instantiated: {model.__class__.__name__}")
            
            # Print model statistics
            print(f"\nTotal parameters: {count_parameters(model):,}")
            
            # Print model structure
            print("\n=== Model Structure ===")
            structure = get_model_structure(model)
            for name, info in structure.items():
                print(f"{name}: {info['type']} - {info.get('parameters', 0):,} parameters")
                if "num_layers" in info:
                    print(f"  - {info['num_layers']} layers")
            
            # Try a forward pass
            print("\n=== Attempting Forward Pass ===")
            dummy_input = create_dummy_input(model_config)
            
            # Set model to eval mode to avoid training-specific operations
            model.eval()
            
            try:
                with torch.no_grad():
                    # This might fail due to missing CUDA extensions
                    outputs = model(dummy_input)
                print("Forward pass successful!")
                print(f"Output shapes: {[k + ': ' + str(v.shape) for k, v in outputs.items() if isinstance(v, torch.Tensor)]}")
            except Exception as e:
                print(f"Forward pass failed: {e}")
                print("This is expected on CPU or without required CUDA extensions.")
        
        except Exception as e:
            print(f"Error instantiating model: {e}")
            
            # Fall back to source code inspection
            print("\n=== Falling back to source code inspection ===")
            hrm_file_path = os.path.join(SAPIENT_HRM_DIR, "models", "hrm", "hrm_act_v1.py")
            if os.path.exists(hrm_file_path):
                with open(hrm_file_path, 'r') as f:
                    hrm_source = f.read()
                
                # Extract class definitions
                class_defs = [line for line in hrm_source.split('\n') if line.startswith('class ')]
                print("HRM Classes defined:")
                for class_def in class_defs:
                    print(f"  {class_def}")
    
    # Step 3: Document findings about architecture
    print("\n=== Architecture Analysis for Adaptation ===")
    print("""
Key Findings:
1. Two-level Hierarchical Architecture:
   - High-level module (H_level): Handles abstract planning with slower cycles
   - Low-level module (L_level): Handles detailed execution with faster cycles

2. Cycle Structure:
   - H_cycles: Number of high-level planning iterations
   - L_cycles: Number of low-level execution iterations per high-level cycle
   - The model uses nested loops of these cycles for deep computation

3. Input Processing:
   - Uses token embeddings + optional puzzle embeddings
   - Supports both learned positional embeddings and rotary embeddings

4. Attention Mechanism:
   - Non-causal attention (bidirectional) in both levels
   - Uses RMSNorm and SwiGLU activations similar to modern LLMs

5. Adaptation Considerations:
   - Replace 2D puzzle inputs with 1D token sequences
   - Modify position encodings for sequence modeling
   - Adapt output head for autoregressive generation
   - Consider making attention causal for code generation
   - Replace puzzle-specific evaluation with Pass@k metrics
    """)
    
    # Step 4: Save findings to a file
    findings_path = os.path.join(os.path.dirname(__file__), "ARCHITECTURE_FINDINGS.md")
    with open(findings_path, 'w') as f:
        f.write("""# Sapient HRM Architecture Analysis

## Overview
The Hierarchical Reasoning Model (HRM) uses a two-level recurrent architecture with a high-level planner and low-level executor. This document captures key architectural details for adaptation to code generation.

## Key Components

### Hierarchical Structure
- **H_level**: High-level planning module operating on slower timescale
- **L_level**: Low-level execution module operating on faster timescale
- **Cycles**: H_cycles (outer) and L_cycles (inner) control computational depth

### Model Parameters
- Hidden size: {hidden_size} dimensions
- Attention heads: {num_heads}
- Layers: {H_layers} in H_level, {L_layers} in L_level
- Parameter count: ~27M total

### Input Processing
- Token embeddings scaled by sqrt(hidden_size)
- Optional puzzle embeddings for task-specific information
- Position encodings: Rotary (RoPE) or learned

### Computation Flow
1. Input embeddings combined with puzzle embeddings
2. Nested loop of H_cycles and L_cycles:
   - L_level processes token-level details using H_level context
   - H_level updates abstract plan based on L_level results
3. Final output projected to vocabulary logits

## Adaptation Strategy for Code Generation

1. **Input Representation**:
   - Replace 2D puzzle grid with 1D token sequences
   - Use standard NLP tokenizer (e.g., GPT-2 BPE)

2. **Position Encoding**:
   - Maintain RoPE for sequence modeling
   - Adjust sequence length for code context

3. **Attention Mechanism**:
   - Consider changing to causal attention for autoregressive generation
   - Maintain bidirectional if using for code completion/infilling

4. **Output Layer**:
   - Adapt for next-token prediction
   - Implement autoregressive sampling for generation

5. **Training Objective**:
   - Replace puzzle-specific loss with standard language modeling loss
   - Add Pass@k evaluation metrics

## Technical Challenges

- CUDA extensions required for full training
- CPU fallback needed for development
- Gradient stability during training

""".format(
    hidden_size=arch_config.get("hidden_size", 512) if combined_config else "512",
    num_heads=arch_config.get("num_heads", 8) if combined_config else "8",
    H_layers=arch_config.get("H_layers", 4) if combined_config else "4",
    L_layers=arch_config.get("L_layers", 4) if combined_config else "4"
))
    
    print(f"\nFindings saved to: {findings_path}")
    print("\nPhase 1 of HRM adaptation plan completed!")

if __name__ == "__main__":
    main()
