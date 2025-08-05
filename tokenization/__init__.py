"""
Tokenization module for HRM code generation.

This module wraps the HuggingFace GPT-2 tokenizer for use in code generation tasks,
providing convenience methods for encoding/decoding, handling special tokens,
and preparing inputs for training and generation.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "tokenizer")
os.makedirs(_CACHE_DIR, exist_ok=True)

# Global tokenizer instance
_tokenizer = None


def get_tokenizer(force_reload: bool = False) -> PreTrainedTokenizer:
    """
    Returns the GPT-2 tokenizer, loading it if necessary.
    
    Args:
        force_reload: If True, reload the tokenizer even if cached
        
    Returns:
        The configured GPT-2 tokenizer
    """
    global _tokenizer
    
    if _tokenizer is None or force_reload:
        # Check for cached tokenizer
        if os.path.exists(os.path.join(_CACHE_DIR, "tokenizer.json")) and not force_reload:
            _tokenizer = AutoTokenizer.from_pretrained(_CACHE_DIR)
        else:
            # Load from HuggingFace and save locally
            _tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            # Set up special tokens - GPT-2 uses the same token for bos/eos/pad
            _tokenizer.pad_token = _tokenizer.eos_token
            
            _tokenizer.save_pretrained(_CACHE_DIR)
            
    return _tokenizer


def encode(
    text: Union[str, List[str]],
    add_bos: bool = False,
    add_eos: bool = False,
    max_length: Optional[int] = None,
    padding: Union[bool, str] = True,
    truncation: bool = True,
    return_tensors: Optional[str] = None,
) -> Dict[str, Union[List[int], torch.Tensor]]:
    """
    Encodes text using the GPT-2 tokenizer.
    
    Args:
        text: Input text or list of texts
        add_bos: Whether to add the beginning-of-sequence token
        add_eos: Whether to add the end-of-sequence token
        max_length: Maximum sequence length (will truncate if needed)
        padding: Type of padding strategy, or whether to pad
        truncation: Whether to truncate sequences to max_length
        return_tensors: Output format ('pt' for PyTorch tensors, None for lists)
        
    Returns:
        Dictionary with 'input_ids' and optionally 'attention_mask'
    """
    tokenizer = get_tokenizer()
    
    # Resolve padding strategy -------------------------------------------------
    #  • If caller passed an explicit string/False we keep it.
    #  • If caller passed boolean `True` we choose a sensible default:
    #       ‣ "max_length" when `max_length` is provided
    #       ‣ "longest"   otherwise
    if isinstance(padding, bool):
        if padding:
            padding_strategy: Union[bool, str] = "max_length" if max_length else "longest"
        else:
            padding_strategy = False
    else:
        # caller already provided string strategy ("max_length", "longest") or similar
        padding_strategy = padding

    # Handle single strings
    is_single_text = isinstance(text, str)
    if is_single_text:
        text = [text]
    
    # Encode with appropriate special tokens
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        padding=padding_strategy,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
    )
    
    # Manually add BOS/EOS if requested
    if add_bos or add_eos:
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            # Handle tensor case
            batch_size = input_ids.size(0)
            if add_bos:
                bos_tensor = torch.full((batch_size, 1), tokenizer.bos_token_id, 
                                        dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([bos_tensor, input_ids], dim=1)
            if add_eos:
                eos_tensor = torch.full((batch_size, 1), tokenizer.eos_token_id, 
                                        dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([input_ids, eos_tensor], dim=1)
            encoded["input_ids"] = input_ids
        else:
            # Handle list case
            for i in range(len(input_ids)):
                if add_bos:
                    input_ids[i] = [tokenizer.bos_token_id] + input_ids[i]
                if add_eos:
                    input_ids[i] = input_ids[i] + [tokenizer.eos_token_id]
    
    # If single text was passed, and we're not returning tensors, unwrap the batch
    if is_single_text and return_tensors is None:
        encoded = {k: v[0] for k, v in encoded.items()}
    
    return encoded


def create_training_batch(
    prompts: List[str],
    completions: List[str],
    max_length: int = 512,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Creates a training batch with shifted labels for causal language modeling.
    
    Args:
        prompts: List of prompt texts
        completions: List of completion texts (solutions)
        max_length: Maximum sequence length
        return_tensors: Output format ('pt' for PyTorch tensors)
        
    Returns:
        Dictionary with 'input_ids' and 'labels' tensors
    """
    tokenizer = get_tokenizer()
    batch_size = len(prompts)
    
    # Prepare input_ids and labels arrays
    input_ids = []
    labels = []
    
    for prompt, completion in zip(prompts, completions):
        # Encode prompt and completion separately
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
        
        # Combine for full sequence
        combined_tokens = prompt_tokens + completion_tokens
        
        # Truncate if too long
        if len(combined_tokens) > max_length:
            combined_tokens = combined_tokens[:max_length]
        
        # Create labels: -100 for prompt (ignored in loss calculation)
        # and shifted sequence for completion
        prompt_len = min(len(prompt_tokens), max_length)
        seq_labels = [-100] * prompt_len
        
        # Add completion labels (shifted for next-token prediction)
        remaining_length = max_length - prompt_len
        completion_len = min(len(completion_tokens), remaining_length)
        
        # For completion tokens, each label is the next token
        for i in range(completion_len - 1):
            seq_labels.append(completion_tokens[i + 1])
        
        # Last completion token predicts EOS or padding
        if completion_len > 0:
            if prompt_len + completion_len < max_length:
                seq_labels.append(tokenizer.eos_token_id)
            else:
                seq_labels.append(-100)  # No target for last token if at max_length
        
        # Pad sequences to max_length
        padding_length = max_length - len(combined_tokens)
        if padding_length > 0:
            combined_tokens.extend([tokenizer.pad_token_id] * padding_length)
            seq_labels.extend([-100] * padding_length)
        
        # Ensure labels are exactly max_length
        seq_labels = seq_labels[:max_length]
        
        input_ids.append(combined_tokens)
        labels.append(seq_labels)
    
    # Convert to tensors if requested
    if return_tensors == "pt":
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
    
    return {
        "input_ids": input_ids,
        "labels": labels
    }


def decode(
    token_ids: Union[List[int], torch.Tensor],
    skip_special_tokens: bool = True,
) -> str:
    """
    Decodes token IDs back to text.
    
    Args:
        token_ids: List or tensor of token IDs
        skip_special_tokens: Whether to skip special tokens in the decoded output
        
    Returns:
        Decoded text
    """
    tokenizer = get_tokenizer()
    
    # Convert tensor to list if needed
    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() > 1:
            # Handle batch dimension
            token_ids = token_ids[0].tolist()
        else:
            token_ids = token_ids.tolist()
    elif isinstance(token_ids[0], list):
        # Handle batch inputs (first element only)
        token_ids = token_ids[0]
        
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def get_vocab_size() -> int:
    """Returns the vocabulary size of the tokenizer."""
    return get_tokenizer().vocab_size


def validate_tokens(token_ids: List[int]) -> bool:
    """
    Validates that all tokens are within the vocabulary.
    
    Args:
        token_ids: List of token IDs to validate
        
    Returns:
        True if all tokens are valid, False otherwise
    """
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    return all(0 <= token_id < vocab_size for token_id in token_ids)


# Export main functions
__all__ = [
    "get_tokenizer",
    "encode",
    "decode",
    "create_training_batch",
    "get_vocab_size",
    "validate_tokens",
]
