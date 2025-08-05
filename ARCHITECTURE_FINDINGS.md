# Sapient HRM Architecture Analysis

## Overview
The Hierarchical Reasoning Model (HRM) uses a two-level recurrent architecture with a high-level planner and low-level executor. This document captures key architectural details for adaptation to code generation.

## Key Components

### Hierarchical Structure
- **H_level**: High-level planning module operating on slower timescale
- **L_level**: Low-level execution module operating on faster timescale
- **Cycles**: H_cycles (outer) and L_cycles (inner) control computational depth

### Model Parameters
- Hidden size: 512 dimensions
- Attention heads: 8
- Layers: 4 in H_level, 4 in L_level
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

